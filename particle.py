import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
from scipy.spatial import KDTree
from probutils import get_random_combination, stochastic_universal_sample, do_KL
import obsgl

def propagate_particles(W, proposal_idxs, states, pd, rejection_sample=False):
    """
    For each particle, sample from the proposal distribution

    Parameters
    ----------
    W: ndarray(M, N)
        STFT magnitudes in the corpus
    proposal_idxs: ndarray(<= N)
        Subset of indices of W to choose more frequently
        based on observations
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
                    if len(proposal_idxs) == 0 or np.random.rand() < 0.25:
                        # Choose a random element not equal to state[j] or state[j]+1
                        next = np.random.randint(N-2)
                    else:
                        # Choose a random element from the proposal set
                        next = np.random.choice(proposal_idxs)
                    # Ex) N = 10, state[j] = 6; avoid 6 and 7
                    if next == states[i][j]: # next = 6, make next 8
                        next = N-2
                    if next == states[i][j]+1: # next 7, make next 9
                        next = N-1
                    state_next[j] = next
            if not rejection_sample or (len(np.unique(state_next)) == p):
                finished = True
                states[i, :] = state_next

def get_particle_musaic_activations(V, W, p, pfinal, pd, sigma, L, P, gamma=0, r=3, neff_thresh=0, use_gpu=True):
    """

    Parameters
    ----------
    V: ndarray(M, T)
        A M x T nonnegative target matrix
    W: ndarray(M, N)
        STFT magnitudes in the corpus
    p: int
        Sparsity parameter for particles
    pfinal: int
        Sparsity parameter for final activations
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
    r: int
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

    ## Setup KDTree for proposal indices
    WMag = np.sqrt(np.sum(W.T**2, axis=1))
    WMag[WMag == 0] = 1
    WNorm = W.T/WMag[:, None] # Vector normalized version for KDTree
    tree = KDTree(WNorm)
    d = 2*(1-gamma)**0.5 # KDTree distance corresponding to gamma cosine similarity
    proposal_idxs = []

    ## Setup W and the observation probability function
    T = V.shape[1]
    N = W.shape[1]
    WDenom = np.sum(W, axis=0)
    WDenom[WDenom == 0] = 1
    observer = obsgl.Observer(p, W/WDenom, V, L, sigma)

    ## Choose initial combinations
    states = [get_random_combination(N, p) for i in range(P)]
    states = np.array(states, dtype=int)
    ws = np.ones(P)/P
    H = np.zeros((N, T))
    chosen_idxs = np.zeros((pfinal, T), dtype=int)
    neff = np.zeros(T)
    wsmax = np.zeros(T)
    for t in range(T):
        if t%10 == 0:
            print(".", end="")
        ## Step 1: Figure out valid indices and collect the columns
        ## of W that correspond to them
        

        ## Step 1: Sample from the proposal distribution
        Vt = V[:, t][:, None]
        propagate_particles(W, proposal_idxs, states, pd)

        ## Step 2: Apply the observation probability updates
        if use_gpu:
            ws *= observer.observe(states, t)
        else:
            ws *= observer.observe_slow(states, t)

        ## Step 3: Figure out the activations for this timestep
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
        for dc in range(max(t-r, 0), t):
            probs[chosen_idxs[:, dc]] = 0
        top_idxs = np.argpartition(-probs, pfinal)[0:pfinal]
        
        chosen_idxs[:, t] = top_idxs
        H[top_idxs, t] = do_KL(W[:, top_idxs], V[:, t], L)
        
        ## Step 4: Resample particles
        ws /= np.sum(ws)
        neff[t] = 1/np.sum(ws**2)
        if neff[t] < neff_thresh:
            choices, _ = stochastic_universal_sample(ws, len(ws))
            states = states[choices, :]
            ws = np.ones(ws.size)/ws.size

        ## Step 5: Sample proposal indices for next iteration based on 
        ## current observation
        Vt = Vt.flatten()
        VtNorm = np.sqrt(np.sum(Vt**2))
        proposal_idxs = []
        if VtNorm > 0:
            proposal_idxs = np.array(tree.query_ball_point(Vt/VtNorm, d), dtype=int)+1
            proposal_idxs = proposal_idxs[proposal_idxs < N]

    return H, wsmax, neff