import numpy as np
from scipy.spatial import KDTree
from probutils import stochastic_universal_sample, do_KL
from observer import Observer
from propagator import Propagator

def get_particle_musaic_activations(V, W, p, pfinal, pd, temperature, L, P, gamma=0, r=3, neff_thresh=0, use_gpu=True):
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
    temperature: float
        Amount to focus on matching observations
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
    tree = None
    if gamma > 0:
        tree = KDTree(WNorm)
    d = 2*(1-gamma)**0.5 # KDTree distance corresponding to gamma cosine similarity
    proposal_idxs = []

    ## Setup W and the observation probability function
    T = V.shape[1]
    N = W.shape[1]
    WDenom = np.sum(W, axis=0)
    WDenom[WDenom == 0] = 1
    observer = Observer(p, W/WDenom, L)
    propagator = Propagator(N, pd)

    ## Choose initial combinations
    states = np.random.randint(N, size=(P, p))
    ws = np.ones(P)/P
    H = np.zeros((N, T))
    chosen_idxs = np.zeros((pfinal, T), dtype=int)
    neff = np.zeros(T)
    wsmax = np.zeros(T)
    for t in range(T):
        if t%10 == 0:
            print(".", end="", flush=True)

        ## Step 1: Sample from the proposal distribution
        Vt = V[:, t][:, None]
        #propagate_particles(W, proposal_idxs, states, pd)
        propagator.propagate_numba(states)

        ## Step 2: Apply the observation probability updates
        dots = []
        if use_gpu:
            dots = observer.observe(states, V[:, t])
        else:
            dots = observer.observe_cpu(states, t)
        obs_prob = np.exp(dots*temperature/np.max(dots))
        obs_prob /= np.sum(obs_prob)
        ws *= obs_prob

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
        if gamma > 0:
            Vt = Vt.flatten()
            VtNorm = np.sqrt(np.sum(Vt**2))
            proposal_idxs = []
            if VtNorm > 0:
                proposal_idxs = np.array(tree.query_ball_point(Vt/VtNorm, d), dtype=int)+1
                proposal_idxs = proposal_idxs[proposal_idxs < N]

    return H, wsmax, neff