import numpy as np
import torch

DB_MIN = -1000

hann_window = lambda N: 0.5*(1 - np.cos(2*np.pi*np.arange(N)/N))

def blackman_harris_window(N):
    """
    Create a Blackman-Harris Window
    
    Parameters
    ----------
    N: int
        Length of window
    
    Returns
    -------
    np.tensor(N): Samples of the window
    """
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    t = np.arange(N)/N
    return a0 - a1*np.cos(2*np.pi*t) + a2*np.cos(4*np.pi*t) - a3*np.cos(6*np.pi*t)

def tri_window(N):
    h = N//2
    ret = np.zeros(N)
    ret[0:h] = np.linspace(0, 1, h)
    ret[h:2*h] = np.linspace(1, 0, h)
    return ret

def get_windowed(x, hop, win, win_fn=hann_window, dc_normalize=True):
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
    dc_normalize: bool
        If True, normalize the windows to sum to 0 after windowing

    Returns
    -------
    ndarray(win, n_windows)
        Windowed audio
    ndarray(n_windows)
        Power of each window
    """
    nwin = int(np.ceil((x.size-win)/hop))+1
    S = np.zeros((win, nwin))
    for j in range(nwin):
        xj = x[hop*j:hop*j+win]
        S[0:xj.size, j] = xj
    power = np.sum(S**2, axis=0)/win
    ind = (power == 0)
    power[ind == 1] = 1
    power = 10*np.log10(power)
    power[ind == 1] = DB_MIN
    S = S*win_fn(win)[:, None]
    if dc_normalize:
        S -= np.mean(S, axis=0, keepdims=True)
    return S, power

def do_windowed_sum(WSound, H, win, hop):
    """
    The inverse of the get_windowed method on a product W*H

    Parameters
    ----------
    WSound: torch.tensor(win_length, K) 
        An win x K matrix of template sounds in some time order along the second axis
    H: torch.tensor(K, N)
        Activations matrix
    win: int
        Window length
    hop: int
        Hop length
    """
    yh = torch.matmul(WSound, H)
    y = torch.zeros(yh.shape[1]*hop+win).to(yh)
    for j in range(yh.shape[1]):
        y[j*hop:j*hop+win] += yh[:, j]
    return y

def load_corpus(path, sr, stereo, amp_normalize=True):
    """
    Load a corpus of audio

    Parameters
    ----------
    path: string
        Path to folder or file
    sr: int
        Sample rate to use
    stereo: bool
        If true, load stereo.  If false, load mono
    amp_normalize: bool
        If True (default), normalize the audio sample range to be in [-1, 1]
    
    Returns
    -------
    ndarray(n_channels, n_samples)
        The audio samples (leave as numpy so the user can choose
        the right torch types later)
    """
    import glob
    import os
    import librosa
    from warnings import filterwarnings
    filterwarnings("ignore", message="librosa.core.audio.__audioread_load*")
    filterwarnings("ignore", message="PySoundFile failed.*")
    samples = []
    files = [path]
    if os.path.isdir(path):
        files = glob.glob(path + "/*")
    N = 0
    for f in sorted(files):
        try:
            x, sr = librosa.load(f, sr=sr, mono=not stereo)
            if amp_normalize:
                norm = np.max(np.abs(x))
                if norm > 0:
                    x = x/norm
            if stereo and len(x.shape) == 1:
                x = np.array([x, x])
            if stereo:
                N += x.shape[1]
            else:
                N += x.size
            print("Finished {}, length {}".format(f, N/sr))
            samples.append(x)
        except:
            
            pass
    if len(samples) == 0:
        print("Error: No usable files found at ", path)
    assert(len(samples) > 0)
    if stereo:
        x = np.concatenate(samples, axis=1)
    else:
        x = np.concatenate(samples)
    if len(x.shape) == 1:
        x = x[None, :]
    return x
