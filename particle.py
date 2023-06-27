import numpy as np
import matplotlib.pyplot as plt
import time
from combinadics import Combination, Choose
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

#@jit(nopython=True)
def particle_musaic_iter(W, WDenom, Ht, Vt, ws, states, states_next, pd, N, sigma, L):
    Vt_std = np.std(Vt)
    P = states.shape[0]
    p = states.shape[1]
    Wd = np.zeros((p, 1))
    hi = np.zeros((p, 1))
    Vi = np.zeros((Vt.shape[0], P))
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

        ## Step 2: Apply observation update
        Wi = W[:, states[i]]
        Wd[:, 0] = WDenom[states[i]]
        hi[:, 0] = Ht[:, i]
        
        for l in range(L):
            WH = Wi.dot(hi)
            WH[WH == 0] = 1
            VLam = Vt/WH
            hi = hi*((Wi.T).dot(VLam)/Wd)
        Ht[:, i] = hi[:, 0]
        Vi[:, i] = Wi.dot(hi).flatten()
    
    num = (np.mean(Vt*np.log(Vt/Vi) - Vt + Vi, axis=0)/Vt_std)**2
    return ws*np.exp(-num/sigma**2)
    

def get_particle_musaic_activations(V, W, p, pd, sigma, L, P, gamma=0.1):
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

    ## Setup KDTree
    WDenom = np.sqrt(np.sum(W.T**2, axis=1))
    WDenom[WDenom == 0] = 1
    WNorm = W.T/WDenom[:, None]
    tree = KDTree(WNorm)
    ## TODO Later: Sample only from states that are close to the kdtree

    T = V.shape[1]
    N = W.shape[1]
    print("T = ", T, ", N = ", N)
    WDenom = np.sum(W, 0)
    WDenom[WDenom == 0] = 1

    ## Choose initial combinations
    c = Combination(N, p)
    states = get_random_combination(Choose(N, p), P)
    print(len(np.unique(states)), "particles")
    states = [c.Element(s).data for s in states]
    states = np.array(states, dtype=int)
    states_next = np.zeros_like(states)
    ws = np.ones(P)/P
    H = np.zeros((N, T))
    idxsmax = np.zeros(T)
    wsmax = np.zeros(T)
    for t in range(T):
        if t%10 == 0:
            print(".", end="")
        Vt = V[:, t][:, None]
        Ht = np.random.rand(p, P)

        ws = particle_musaic_iter(W, WDenom, Ht, Vt, ws, states, states_next, pd, N, sigma, L)

        ## Step 3: Resample
        ws /= np.sum(ws)
        ## TODO: Finish this
        idx = np.argmax(ws)
        H[states[idx], t] = Ht[:, idx]
        idxsmax[t] = idx
        wsmax[t] = ws[idx]

    
    return H, idxsmax, wsmax