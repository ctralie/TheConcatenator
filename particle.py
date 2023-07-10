import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
from scipy.spatial import KDTree
from probutils import get_random_combination, stochastic_universal_sample, do_KL
import obsgl

def update_valid_idxs(valid_idxs, tree, Vt, d, B, N):
    """
    Come up with an updated list of pruned indices in W

    Parameters
    ----------
    valid_idxs: deque of list
        A deque with all of the valid incides up to the past B timesteps.
        This is updated as a side effect of the method
    tree: KDTree
        KDTree into normalized elements of W
    Vt: ndarray(M)
        Current observation
    d: float
        Cutoff distance in kdtree
    B: int
        Length of block in which to consider observations
    N: int
        Number of columns in W
    
    Returns
    -------
    pruned_idxs: ndarray(<=N)
        Pruned set of indices
    """
    # First move forward each index in time from before
    for v in valid_idxs:
        if len(v) > 0:
            v[:] += 1
    # Now add on the new indices from this observation
    VtNorm = np.sqrt(np.sum(Vt**2))
    tidxs = []
    if VtNorm > 0:
        tidxs = np.array(tree.query_ball_point(Vt/VtNorm, d), dtype=int)
    valid_idxs.append(tidxs)
    if len(valid_idxs) > B:
        valid_idxs.popleft()
    # Collect all of the indices together into a unique list
    pruned_idxs = np.array([])
    for idxs in valid_idxs:
        pruned_idxs = np.concatenate((pruned_idxs, idxs))
    pruned_idxs = np.unique(pruned_idxs)
    pruned_idxs = set(pruned_idxs[pruned_idxs < N])
    return pruned_idxs, tidxs

def propagate_particles(W, pruned_idxs, states, pd, rejection_sample=False):
    """
    For each particle, sample from the proposal distribution

    Parameters
    ----------
    W: ndarray(M, N)
        STFT magnitudes in the corpus
    pruned_idxs: ndarray(<= N)
        Subset of indices of W to use based on observation proximity
    states: ndarray(P, p)
        Column choices in W corresponding to each particle;
        Updated by reference
    pd: float
        Probability of remaining in the same column in time order
    rejection_sample: bool
        Whether to do rejection sampling to prevent collisions of activations
    """
    N = W.shape[1]
    P = states.shape[0]
    p = states.shape[1]

    state_next = np.zeros(p, dtype=int)
    for i in range(P):
        ## Step 1: Sample from proposal distribution for each particle
        finished = False
        while not finished: # Do rejection sampling
            for j in range(p):
                if states[i][j] < N-1 and np.random.rand() < pd:
                    state_next[j] = states[i][j] + 1
                else:
                    # Choose a random element not equal to state[j] or state[j]+1
                    next = np.random.randint(N-2)
                    # Ex) N = 10, state[j] = 6; avoid 6 and 7
                    if next == states[i][j]: # next = 6, make next 8
                        next = N-2
                    if next == states[i][j]+1: # next 7, make next 9
                        next = N-1
                    state_next[j] = next
            if not rejection_sample or (len(np.unique(state_next)) == p):
                finished = True
                states[i, :] = state_next
        
        ## Step 2: If pruning is enabled:
        ## For any index that's not in the set of pruned indices,
        ## sample uniformly at random from one of the pruned ones
        if len(pruned_idxs) >= p:
            pruned_copy = pruned_idxs.copy()
            for j in range(p):
                if states[i, j] in pruned_idxs:
                    pruned_copy.remove(states[i, j])
            choices = np.array(list(pruned_copy))
            np.random.shuffle(choices)
            cidx = 0
            for j in range(p):
                if states[i, j] not in pruned_idxs:
                    states[i, j] = choices[cidx]
                    cidx += 1

def get_particle_musaic_activations(V, W, p, pd, sigma, L, P, gamma=0, c=3, neff_thresh=0, use_gpu=True):
    """

    Parameters
    ----------
    V: ndarray(M, T)
        A M x T nonnegative target matrix
    W: ndarray(M, N)
        STFT magnitudes in the corpus
    p: int
        Sparsity parameter
    pd: float
        State transition probability
    sigma: float
        Observation variance
    L: int
        Number of iterations for NMF observation probabilities
    P: int
        Number of particles
    gamma: float
        Cosine similarity cutoff
    c: int
        Repeated activations cutoff
    neff_thresh: float
        Number of effective particles below which to resample
    use_gpu: bool
        Whether to use the gpu for the observation probabilities
    
    Returns
    -------
    H: ndarray(K, N)
        Activations matrix
    wsmax: ndarray(N)
        Probability of maximum weight chosen at each timestep
    neff: ndarray(N)
        Effective number of particles at each timestep
    """

    ## Setup KDTree and block sampler
    WMag = np.sqrt(np.sum(W.T**2, axis=1))
    WMag[WMag == 0] = 1
    WNorm = W.T/WMag[:, None] # Vector normalized version for KDTree
    tree = KDTree(WNorm)
    B = 2*int(np.sqrt(pd)/(1-pd)) # Block length
    d = 2*(1-gamma)**0.5 # KDTree distance corresponding to gamma cosine similarity
    valid_idxs = deque() # Deque of sets of valid indices, indexed by block

    ## Setup W and the observation probability function
    T = V.shape[1]
    N = W.shape[1]
    print("T = ", T, ", N = ", N)
    WDenom = np.sum(W, axis=0)
    WDenom[WDenom == 0] = 1
    observer = obsgl.Observer(p, W/WDenom, V, L, sigma)

    ## Choose initial combinations
    states = [get_random_combination(N, p) for i in range(P)]
    states = np.array(states, dtype=int)
    ws = np.ones(P)/P
    H = np.zeros((N, T))
    chosen_idxs = np.zeros((p, T), dtype=int)
    neff = np.zeros(T)
    wsmax = np.zeros(T)
    for t in range(T):
        if t%10 == 0:
            print(".", end="")
        ## Step 1: Figure out valid indices and collect the columns
        ## of W that correspond to them
        Vt = V[:, t]
        pruned_idxs = []
        if gamma > 0:
            pruned_idxs, tidxs = update_valid_idxs(valid_idxs, tree, Vt, d, B, N)
            #print(N, len(tidxs), len(pruned_idxs))

        ## Step 2: Sample from the proposal distribution
        Vt = Vt[:, None]
        propagate_particles(W, pruned_idxs, states, pd)

        ## Step 3: Apply the observation probability updates
        if use_gpu:
            ws *= observer.observe(states, t)
        else:
            ws *= observer.observe_slow(states, t)

        ## Step 4: Figure out the activations for this timestep
        ## by aggregating multiple particles near the top
        wsmax[t] = np.max(ws)
        probs = np.zeros(N)
        max_particles = np.argpartition(-ws, 2*p)[0:2*p]
        for state, w in zip(states[max_particles], ws[max_particles]):
            probs[state] += w
        # Promote states that follow the last state that was chosen
        for dc in range(max(t-1, 0), t):
            last_state = chosen_idxs[:, dc] + (t-dc)
            probs[last_state[last_state < N]] *= 5
        # Zero out last ones to prevent repeated activations
        for dc in range(max(t-c, 0), t):
            probs[chosen_idxs[:, dc]] = 0
        top_idxs = np.argpartition(-probs, p)[0:p]
        
        chosen_idxs[:, t] = top_idxs
        H[top_idxs, t] = do_KL(W[:, top_idxs], V[:, t], L)
        
        ## Step 5: Resample particles
        ws /= np.sum(ws)
        neff[t] = 1/np.sum(ws**2)
        if neff[t] < neff_thresh:
            choices, _ = stochastic_universal_sample(ws, len(ws))
            print("Resampling, Unique choices:", len(np.unique(choices)))
            states = states[choices, :]
            ws = np.ones(ws.size)/ws.size

    return H, wsmax, neff