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
from numba import jit

def get_activations_diff(H, p):
    """
    Compute the number of activations that differ between
    adjacent time frames

    Parameters
    ----------
    H: ndarray(N, T)
        Activations matrix
    p: int
        Number of activations per timestep
    
    Returns
    -------
    ndarray(T-1)
        Number of activations that are different at each timestep (at most p)
    """
    idx = np.argsort(-H, axis=0)[0:p, :]
    diff = []
    for i in range(idx.shape[1]-1):
        s1 = set(idx[:, i]+1)
        s2 = set(idx[:, i+1])
        same = len(s1.intersection(s2))
        diff.append(p-same)
    return np.array(diff)

def get_col_sets(row, col):
    """
    Parameters
    ----------
    row: ndarray(N)
        Row indices of nonzero activations
    col: ndarray(N)
        Column indices of nonzero activations
    
    Returns
    -------
    list of set
        Set of row indices for each column
    """
    cols = {}
    for i, j in zip(row, col):
        if not j in cols:
            cols[j] = set([i])
        else:
            cols[j].add(i)
    return cols

def get_activations_diff_sparse(row, col, p):
    """
    Compute the number of activations that differ between
    adjacent time frames, given a sparse representation of
    a matrix of activations

    Parameters
    ----------
    row: ndarray(N)
        Row indices of nonzero activations
    col: ndarray(N)
        Column indices of nonzero activations
    p: int
        Number of activations per timestep
    
    Returns
    -------
    ndarray(T-1)
        Number of activations that are different at each timestep (at most p)
    """
    cols = get_col_sets(row, col)
    N = np.max(col)+1
    diff = p*np.ones(N-1)
    for j in range(N-1):
        if j in cols and j+1 in cols:
            s1 = set([x+1 for x in cols[j]])
            diff[j] = p - len(s1.intersection(cols[j+1]))
    return diff

def get_repeated_activation_itervals(H, p):
    """
    Compute the distances from each activation to its first repetition

    Parameters
    ----------
    H: ndarray(N, T)
        Activations matrix
    p: int
        Number of activations per timestep
    
    Returns
    -------
    ndarray(<= p*T)
        Distance of each activation to its first repetition, 
        if that activation actually had a repetition
    """
    idx = np.argsort(-H, axis=0)[0:p, :]
    columns = [set(idx[:, j]) for j in range(H.shape[1])]
    diffs = []
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            k = j+1
            finished = False
            while k < H.shape[1] and not finished:
                if idx[i, j] in columns[k]:
                    finished = True
                else:
                    k += 1
            if finished:
                diffs.append(k-j)
    return np.array(diffs, dtype=int)

def get_diag_lengths(H, p):
    """
    Compute the lengths of all diagonals

    Parameters
    ----------
    H: ndarray(N, T)
        Activations matrix
    p: int
        Number of activations per timestep
    
    Returns
    -------
    ndarray(<= p*T)
        The lengths of each diagonal
    """
    idxs = np.argsort(-H, axis=0)[0:p, :]
    idxs = [set(idxs[:, j]) for j in range(H.shape[1])]
    diags = []
    for j in range(H.shape[1]):
        for idx in idxs[j]:
            k = j+1
            while k < H.shape[1] and idx+(k-j) in idxs[k]:
                k += 1
            diags.append(k-j)
    return np.array(diags, dtype=int)

def get_diag_lengths_sparse(row, col):
    """
    Compute the lengths of all diagonals

    Parameters
    ----------
    row: ndarray(N)
        Row indices of nonzero activations
    col: ndarray(N)
        Column indices of nonzero activations
    
    Returns
    -------
    ndarray(<= p*T)
        The lengths of each diagonal
    """
    cols = get_col_sets(row, col)
    diags = []
    N = len(cols)
    for j in range(N):
        if j in cols:
            for idx in cols[j]:
                k = j+1
                while k < N and k in cols and idx+(k-j) in cols[k]:
                    k += 1
                diags.append(k-j)
    return np.array(diags, dtype=int)

def count_top_activations(states, top_idxs):
    """
    Count how many of the top activations each particle has

    Parameters
    ----------
    states: ndarray(P, p)
        Column choices in W corresponding to each particle
    top_idxs: ndarray(pfinal)
        Final activations that were chosen
    
    Returns
    -------
    ndarray(P)
        Number of top activations in each particle
    """
    P = states.shape[0]
    res = np.zeros(P)
    for idx in top_idxs:
        res += np.sum(states == idx, axis=1)
    return res

def get_random_combination(N, p):
    """
    Choose p elements out of N possible elements in {0, 1, ..., N-1}
    using the Fisher-Yates algorithm, without allocating an array of length N

    Parameters
    ----------
    N: int
        The options in {0, 1, 2, ..., N-1}
    p: int
        The number of choices to make
    
    Returns
    -------
    ndarray(p, dtype=int)
        The array of choices
    """
    from scipy import sparse
    # Only store elements that are not equal to their index
    s = sparse.coo_matrix(([], ([], [])), shape=(1, N)).tolil()
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

@jit(nopython=True)
def stochastic_universal_sample(ws, target_points):
    """
    Resample indices according to universal stochastic sampling.
    Also known as "systematic resampling"
    [1] G. Kitagawa, "Monte Carlo filter and smoother and non-Gaussian nonlinear
    state space models," J. Comput. Graph. Stat., vol. 5, no. 1, pp. 1-25, 1996.
    [2] J. Carpenter, P. Clifford, and P. Fearnhead, "An improved particle filter [...]"

    Parameters
    ----------
    ndarray(P)
        The normalized weights of the particles
    target_points: int
        The number of desired samples
    
    Returns
    -------
    ndarray(P, dtype=int)
        Indices of sampled particles, with replacement
    ndarray(P)
        Weights of the new samples
    """
    counts = np.zeros(ws.size)
    w = np.zeros(ws.size+1)
    order = np.random.permutation(ws.size)
    w[1::] = ws.flatten()[order]
    w = np.cumsum(w)
    p = np.random.rand() # Cumulative probability index, start off random
    idx = 0
    for i in range(target_points):
        while not (p >= w[idx] and p < w[idx+1]):
            idx += 1
        idx = min(idx, ws.size-1) # Safeguard
        counts[order[idx]] += 1
        p += 1/target_points
        if p >= 1:
            p -= 1
            idx = 0
    ws_new = np.zeros(ws.size)
    choices = np.zeros(ws.size)
    idx = 0
    for i in range(len(counts)):
        for w in range(int(counts[i])):
            choices[idx] = i
            ws_new[idx] = ws[i]/counts[i]
            idx += 1
    ws_new /= np.sum(ws_new)
    return choices, ws_new

def do_KL(Wi, WAlpha, Vt, L):
    """
    Perform a KL-based NMF

    Parameters
    ----------
    Wi: ndarray(M, p)
        Templates
    WAlpha: ndarray(p)
        L2 penalties for each activation
    Vt: ndarray(M)
        Observation
    L: int
        Number of iterations
    
    Returns
    -------
    h: ndarray(p)
        Activations
    """
    hi = np.random.rand(Wi.shape[1])
    #hi = np.ones(Wi.shape[1])
    Wd = np.sum(Wi, axis=0)
    Wd[Wd == 0] = 1
    for _ in range(L):
        WH = Wi.dot(hi)
        WH[WH == 0] = 1
        VLam = Vt/WH
        hi *= ((Wi.T).dot(VLam)/(Wd + WAlpha*hi))
    return hi

def do_KL_torch(Wi, WAlpha, Vt, L):
    """
    Perform a KL-based NMF using pytorch

    Parameters
    ----------
    Wi: torch.tensor(M, p)
        Templates
    WAlpha: torch.tensor(p)
        L2 penalties for each activation
    Vt: torch.tensor(M)
        Observation
    L: int
        Number of iterations
    
    Returns
    -------
    h: ndarray(p)
        Activations
    """
    import torch
    hi = torch.rand(Wi.shape[1]).to(Wi)
    Wd = torch.sum(Wi, dim=0)
    Wd[Wd == 0] = 1
    WiT = torch.movedim(Wi, 0, 1)
    for _ in range(L):
        WH = torch.matmul(Wi, hi)
        WH[WH == 0] = 1
        VLam = Vt/WH
        hi *= torch.matmul(WiT, VLam)/(Wd + WAlpha*hi)
    return hi
