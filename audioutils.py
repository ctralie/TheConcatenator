import numpy as np

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
    ndarray(N) if mono, ndarray(2, N) if stereo
        The audio samples
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
    for f in files:
        try:
            x, sr = librosa.load(f, sr=sr, mono=not stereo)
            if stereo and len(x.shape) == 1:
                x = np.array([x, x])
            samples.append(x)
        except:
            pass
    if stereo:
        x = np.concatenate(samples, axis=1)
    else:
        x = np.concatenate(samples)
    return x
