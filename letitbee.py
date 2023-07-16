"""
Purpose: To implementing the NMF techniques in [1]
[1] Driedger, Jonathan, Thomas Praetzlich, and Meinard Mueller. 
"Let it Bee-Towards NMF-Inspired Audio Mosaicing." ISMIR. 2015.
"""
import numpy as np
import scipy.io as sio
import scipy.ndimage
import matplotlib.pyplot as plt
import time
import librosa
import librosa.display

hann_window = lambda N: 0.5*(1 - np.cos(2*np.pi*np.arange(N)/N))

def get_windowed(x, hop, win, win_fn=hann_window):
    """
    Stack sliding windows of audio, multiplied by a window function,
    into the columns of a 2D array

    x: ndarray(N)
        Audio clip of N samples
    win: int
        Window length
    hop: int
        Hop length
    win_fn: int -> ndarray(win)
        Window function

    Returns
    -------
    ndarray(win, n_windows)
        Windowed audio
    """
    nwin = int(np.ceil((x.size-win)/hop))+1
    S = np.zeros((win, nwin))
    coeff = win_fn(win)
    for j in range(nwin):
        xj = x[hop*j:hop*j+win]
        # Zeropad if necessary
        if len(xj) < win:
            xj = np.concatenate((xj, np.zeros(win-len(xj))))
        # Apply window function
        xj = coeff*xj
        S[:, j] = xj
    return S


def do_windowed_sum(WSound, H, win, hop):
    """
    The inverse of the get_windowed method on a product W*H

    Parameters
    ----------
    WSound: ndarray(win_length, K) 
        An win x K matrix of template sounds in some time order along the second axis
    H: ndarray(K, N)
        Activations matrix
    win: int
        Window length
    hop: int
        Hop length
    """
    yh = WSound.dot(H)
    y = np.zeros(yh.shape[1]*hop+win)
    for j in range(yh.shape[1]):
        y[j*hop:j*hop+win] += yh[:, j]
    return y


def diagonally_enhance_H(H, c):
    """
    Diagonally enhance an activation matrix in place

    Parameters
    ----------
    H: ndarray(N, K)
        Activation matrix
    c: int
        Diagonal half-width
    """
    K = H.shape[0]
    di = K-1
    dj = 0
    for k in range(-H.shape[0]+1, H.shape[1]):
        z = np.cumsum(np.concatenate((np.zeros(c+1), np.diag(H, k), np.zeros(c))))
        x2 = z[2*c+1::] - z[0:-(2*c+1)]
        H[di+np.arange(len(x2)), dj+np.arange(len(x2))] = x2
        if di == 0:
            dj += 1
        else:
            di -= 1

def get_musaic_activations(V, W, L, r=3, p=10, c=3, verbose=False):
    """
    Implement the technique from "Let It Bee-Towards NMF-Inspired
    Audio Mosaicing"

    Parameters
    ----------
    V: ndarray(M, N)
        A M x N nonnegative target matrix
    W: ndarray(M, K)
        STFT magnitudes corresponding to WSound
    L: int
        Number of iterations
    r: int
        Width of the repeated activation filter
    p: int
        Degree of polyphony; i.e. number of values in each column of H which should be 
        un-shrunken
    c: int
        Half length of time-continuous activation filter
    verbose: bool
        Whether to print progress info
    
    Returns
    -------
    H: ndarray(K, N)
        Activations matrix
    """
    N = V.shape[1]
    K = W.shape[1]
    WDenom = np.sum(W, 0)
    WDenom[WDenom == 0] = 1

    VAbs = np.abs(V)
    H = np.random.rand(K, N)
    for l in range(L):
        if verbose:
            print(l, end='.') # Print out iteration number for progress
        iterfac = 1-float(l+1)/L       

        #Step 1: Avoid repeated activations
        MuH = scipy.ndimage.filters.maximum_filter(H, size=(1, 2*r+1))
        H[H<MuH] = H[H<MuH]*iterfac

        #Step 2: Restrict number of simultaneous activations
        #Use partitions instead of sorting for speed
        colCutoff = -np.partition(-H, p, 0)[p, :] 
        H[H < colCutoff[None, :]] = H[H < colCutoff[None, :]]*iterfac
        #Step 3: Supporting time-continuous activations
        diagonally_enhance_H(H, c)

        #KL Divergence Version
        WH = W.dot(H)
        WH[WH == 0] = 1
        VLam = VAbs/WH
        H = H*((W.T).dot(VLam)/WDenom[:, None])
    
    return H



def create_musaic_sliding(V, WSound, W, win, hop, slidewin, L, r=3, p=10, c=3):
    nwin = V.shape[1]-slidewin+1
    y = np.zeros(V.shape[1]*hop+win)
    for j in range(nwin):
        print(j, end=".")
        H = get_musaic_activations(V[:, j:j+slidewin], W, L, r, p, c)
        if j == 0:
            yj = do_windowed_sum(WSound, H, win, hop)
            y[0:yj.size] = yj
        else:
            H = H[:, -1::]
            j1 = hop*(j+slidewin-1)
            yj = WSound.dot(H)
            y[j1:j1+win] += yj.flatten()
    return y

