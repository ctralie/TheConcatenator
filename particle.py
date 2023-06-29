import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
from scipy.spatial import KDTree
from numba import jit

def get_random_combination(N, p):
    """
    Choose p elements out of N possible elements in {0, 1, ..., N-1}
    using the Fisher-Yates algorithm, without allocating an array of length N
    """
    from scipy import sparse
    # Only store elements that are not equal to their index
    s = sparse.coo_matrix(([], ([], [])), shape=(1, N)).tocsr()
    for i in range(p):
        idx = np.random.randint(N-i)+i
        if idx != i:
            s_before = s[0, i]
            # Swap s[idx] and s[i]
            if not s[0, idx]:
                s[0, i] = idx
            else:
                s[0, i] = s[0, idx]
            if not s_before:
                s[0, idx] = i
            else:
                s[0, idx] = s_before
    ret = s[0, 0:p].toarray()
    return np.array(ret.flatten(), dtype=int)

def do_KL(Wi, Wd, Vt, hi, L):
    for l in range(L):
        WH = Wi.dot(hi)
        WH[WH == 0] = 1
        VLam = Vt/WH
        hi[:] *= ((Wi.T).dot(VLam)/Wd)

#@jit(nopython=True)
def propagate_particles(W, WDenom, pruned_idxs, Ht, Vt, ws, states, states_next, pd, sigma, L):
    """
    For each particle, sample from the proposal distribution, then update
    its weight by the posterior probability

    Parameters
    ----------
    W: ndarray(M, N)
        STFT magnitudes in the corpus
    WDenom: ndarray(N)
        Sum down the columns of W
    pruned_idxs: ndarray(<= N)
        Subset of indices of W to use based on observation proximity
    Ht: ndarray(p, P)
        Hs for each particle corresponding to the observation (updated by reference)
    Vt: ndarray(M, 1)
        Observation
    ws: ndarray(P)
        Weights of particles
    states: ndarray(P, p)
        Column choices in W corresponding to each particle
    states_next: ndarray(P, p)
        Working memory to hold the states that are being sampled next
    pd: float
        Probability of remaining in the same column in time order
    sigma: float
        Observation variance
    L: int
        Number of iterations for NMF observation probabilities
    """
    N = W.shape[1]
    Vt_std = np.std(Vt)
    P = states.shape[0]
    p = states.shape[1]
    Wd = np.zeros((p, 1))
    hi = np.zeros((p, 1))
    Vi = np.zeros((Vt.shape[0], P))
    pruned_idxs_list = list(pruned_idxs)

    for i in range(P):
        ## Step 1: Sample from proposal distribution for each particle
        finished = False
        while not finished: # Do rejection sampling
            for j in range(p):
                if states[i][j] < N-1 and np.random.rand() < pd:
                    states_next[i][j] = states[i][j] + 1
                else:
                    # Choose a random element not equal to state[j] or state[j]+1
                    next = np.random.randint(N-2)
                    # Ex) N = 10, state[j] = 6; avoid 6 and 7
                    if next == states[i][j]: # next = 6, make next 8
                        next = N-2
                    if next == states[i][j]+1: # next 7, make next 9
                        next = N-1
                    states_next[i][j] = next
            if len(np.unique(states_next[i])) == p:
                finished = True
                states[i, :] = states_next[i, :]
        
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

        ## Step 3: Apply observation update
        Wi = W[:, states[i]]
        Wd[:, 0] = WDenom[states[i]]
        hi[:, 0] = Ht[:, i]
        do_KL(Wi, Wd, Vt, hi, L)
        Ht[:, i] = hi[:, 0]
        Vi[:, i] = Wi.dot(hi).flatten()
    
    num = (np.mean(Vt*np.log(Vt/Vi) - Vt + Vi, axis=0)/Vt_std)**2
    ws[:] *= np.exp(-num/sigma**2)
    

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

def get_marginal_probabilities(states, ws, N):
    """
    Compute the marginal probabilities of each state in W

    Parameters
    ----------
    states: ndarray(P, p)
        Column choices in W corresponding to each particle
    ws: ndarray(P)
        Weights of particles
    N: int
        Number of columns in W
    """
    probs = np.zeros(N)
    for state, w in zip(states, ws):
        probs[state] += w
    return probs / states.shape[1]

def get_particle_musaic_activations(V, W, p, pd, sigma, L, P, gamma=0):
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
    
    Returns
    -------
    H: ndarray(K, N)
        Activations matrix
    """

    ## Setup KDTree and block sampler
    WDenom = np.sqrt(np.sum(W.T**2, axis=1))
    WDenom[WDenom == 0] = 1
    WNorm = W.T/WDenom[:, None]
    tree = KDTree(WNorm)
    B = 2*int(np.sqrt(pd)/(1-pd)) # Block length
    d = 2*(1-gamma)**0.5 # KDTree distance corresponding to gamma cosine similarity
    valid_idxs = deque() # Deque of sets of valid indices, indexed by block

    
    T = V.shape[1]
    N = W.shape[1]
    print("T = ", T, ", N = ", N)
    WDenom = np.sum(W, axis=0)
    WDenom[WDenom == 0] = 1

    ## Choose initial combinations
    states = [get_random_combination(N, p) for i in range(P)]
    states = np.array(states, dtype=int)
    states_next = np.zeros_like(states)
    ws = np.ones(P)/P
    H = np.zeros((N, T))
    HMarginal = np.zeros((N, T))
    probs = np.zeros((N, T))
    idxsmax = np.zeros(T)
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

        ## Step 2 and 3: Sample from proposal distribution and apply observation update
        Vt = Vt[:, None]
        Ht = np.random.rand(p, P)
        propagate_particles(W, WDenom, pruned_idxs, Ht, Vt, ws, states, states_next, pd, sigma, L)
        probs[:, t] = get_marginal_probabilities(states, ws, N)

        ## Step 4: Resample
        ws /= np.sum(ws)
        ## TODO: Finish this
        idx = np.argmax(ws)
        H[states[idx], t] = Ht[:, idx]
        idxsmax[t] = idx
        wsmax[t] = ws[idx]
    
    return H, idxsmax, wsmax, probs