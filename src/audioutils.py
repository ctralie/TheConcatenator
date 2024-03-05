import numpy as np
import torch

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
    torch.tensor(win, n_windows)
        Windowed audio
    """
    nwin = int(np.ceil((x.size-win)/hop))+1
    S = np.zeros((win, nwin))
    for j in range(nwin):
        xj = x[hop*j:hop*j+win]
        S[0:xj.size, j] = xj
    return S*win_fn(win)[:, None]

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

def load_corpus(path, sr, stereo):
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
    for f in files:
        try:
            x, sr = librosa.load(f, sr=sr, mono=not stereo)
            if stereo and len(x.shape) == 1:
                x = np.array([x, x])
            x -= np.mean(x, axis=1, keepdims=True)
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
