import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial import KDTree
from probutils import do_KL

def get_bayes_musaic_activations(V, W, p, pd, sigma, L, gamma=0, c=3):
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
    gamma: float
        Cosine similarity cutoff
    c: int
        Repeated activations cutoff
    
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
    WMag = np.sqrt(np.sum(W**2, axis=0))
    WMag[WMag == 0] = 1
    WNorm = W/WMag[None, :] # Vector normalized version for KDTree
    tree = KDTree(WNorm.T)
    d = 2*(1-gamma)**0.5 # KDTree distance corresponding to gamma cosine similarity

    ## Setup W and the observation probability function
    T = V.shape[1]
    N = W.shape[1]

    ## Initialize weights
    ws = np.ones(N)/N
    H = np.zeros((N, T))
    chosen_idxs = np.zeros((p, T), dtype=int)
    jump_fac = (1-pd)/(N-2)
    neff = np.zeros(T)
    wsmax = np.zeros(T)

    for t in range(T):
        if t%10 == 0:
            print(".", end="")

        ## Step 1: Apply the transition probabilities
        wspad = np.concatenate((ws[-1::], ws))
        ws = pd*wspad[0:-1] + jump_fac*(1-wspad[0:-1]-ws)

        ## Step 2: Apply the observation probability updates
        obs_prob = np.sum(V[:, t][:, None]*WNorm, axis=0)
        obs_prob = np.exp(obs_prob*sigma/np.max(obs_prob))
        denom = np.sum(obs_prob)
        if denom > 0:
            obs_prob /= denom
            ws = ws*obs_prob
            ws /= np.sum(ws)

        ## Step 3: Figure out the activations for this timestep
        ## by aggregating multiple particles near the top
        # Promote states that follow the last state that was chosen
        probs = np.array(ws)
        for dc in range(max(t-1, 0), t):
            last_state = chosen_idxs[:, dc] + (t-dc)
            probs[last_state[last_state < N]] *= 5
        # Zero out last ones to prevent repeated activations
        for dc in range(max(t-c, 0), t):
            probs[chosen_idxs[:, dc]] = 0
        top_idxs = np.argpartition(-probs, p)[0:p]
        
        chosen_idxs[:, t] = top_idxs
        H[top_idxs, t] = do_KL(W[:, top_idxs], V[:, t], L)
        
        neff[t] = 1/np.sum(ws**2)
        wsmax[t] = np.max(ws)
    
    return H, wsmax, neff