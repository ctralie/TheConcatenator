"""
Code copyright Christopher J. Tralie, 2024
Attribution-NonCommercial-ShareAlike 4.0 International


Share — copy and redistribute the material in any medium or format
The licensor cannot revoke these freedoms as long as you follow the license terms.

 Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made . You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    NonCommercial — You may not use the material for commercial purposes .
    NoDerivatives — If you remix, transform, or build upon the material, you may not distribute the modified material.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
"""

import numpy as np
from probutils import do_KL

def get_bayes_musaic_activations(V, W, p, pd, temperature, L, r=3):
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
    temperature: float
        Amount to focus on matching observations
    L: int
        Number of iterations for NMF observation probabilities
    r: int
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
            print(".", end="", flush=True)

        ## Step 1: Apply the transition probabilities
        wspad = np.concatenate((ws[-1::], ws))
        ws = pd*wspad[0:-1] + jump_fac*(1-wspad[0:-1]-ws)

        ## Step 2: Apply the observation probability updates
        obs_prob = np.sum(V[:, t][:, None]*WNorm, axis=0)
        obs_prob = np.exp(obs_prob*temperature/np.max(obs_prob))
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
        for dc in range(max(t-r, 0), t):
            probs[chosen_idxs[:, dc]] = 0
        top_idxs = np.argpartition(-probs, p)[0:p]
        
        chosen_idxs[:, t] = top_idxs
        H[top_idxs, t] = do_KL(W[:, top_idxs], V[:, t], L)
        
        neff[t] = 1/np.sum(ws**2)
        wsmax[t] = np.max(ws)
    
    return H, wsmax, neff