import numpy as np

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
    ndarray(N): Samples of the window
    """
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    t = np.arange(N)/N
    return a0 - a1*np.cos(2*np.pi*t) + a2*np.cos(4*np.pi*t) - a3*np.cos(6*np.pi*t)

def get_mel_filterbank(win_length, sr, pmin, pmax, psub=3, normalize=False):
    """
    Compute a mel-spaced filterbank
    
    Parameters
    ----------
    win_length: int
        Window length (should be around 2*K)
    sr: int
        The sample rate, in hz
    pmin: int
        Note number of lowest note
    pmax: int
        Note number of the highest note
    psub: int
        Number of subdivisions per halfstep
    normalize: bool
        If true, normalize the rows
    
    Returns
    -------
    ndarray(pmax-pmin+1, K)
        The triangular mel filterbank
    """
    K = win_length//2 + 1
    n_bins = (pmax-pmin+1)*psub
    ps = np.linspace(pmin, pmax+1, n_bins+2)
    freqs = 440*(2**(ps/12))
    print("Min Freq: {:.3f}, Max Freq: {:.3f}".format(freqs[0], freqs[-1]))
    bins = freqs*win_length/sr
    bins = np.array(np.round(bins), dtype=int)
    Mel = np.zeros((n_bins, K))
    for i in range(n_bins):
        i1 = bins[i]
        i2 = bins[i+1]
        if i1 == i2:
            i2 += 1
        i3 = bins[i+2]
        if i3 <= i2:
            i3 = i2+1
        tri = np.zeros(K)
        tri[i1:i2] = np.linspace(0, 1, i2-i1)
        tri[i2:i3] = np.linspace(1, 0, i3-i2)
        Mel[i, :] = tri
    if normalize:
        Norm = np.sum(Mel, axis=1)
        Mel = Mel/Norm[:, None]
    return Mel

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
    ndarray(n_channels, n_samples)
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
    if len(x.shape) == 1:
        x = x[None, :]
    return x
