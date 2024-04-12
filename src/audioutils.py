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

DB_MIN = -1000

hann_window = lambda N: (0.5*(1 - np.cos(2*np.pi*np.arange(N)/N))).astype(np.float32)

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

def get_windowed(x, win, win_fn=hann_window):
    """
    Stack sliding windows of audio, multiplied by a window function,
    into the columns of a 2D array

    x: ndarray(N)
        Audio clip of N samples
    win: int
        Window length
    win_fn: int -> ndarray(win)
        Window function

    Returns
    -------
    ndarray(win, n_windows)
        Windowed audio
    ndarray(n_windows)
        Power of each window
    """
    hop = win//2
    nwin = (x.size-win)//hop+1 # Chop off anything that doesn't fit in an even number of windows
    S = np.zeros((win, nwin), dtype=np.float32)
    n_even = x.size//win
    n_odd = nwin - n_even
    S[:, 0::2] = x[0:n_even*win].reshape((n_even, win)).T
    S[:, 1::2] = x[hop:hop+n_odd*win].reshape((n_odd, win)).T
    power = np.sum(S**2, axis=0)/win
    ind = (power == 0)
    power[ind == 1] = 1
    power = 10*np.log10(power)
    power[ind == 1] = DB_MIN
    S = S*win_fn(win)[:, None]
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
    import torch
    yh = torch.matmul(WSound, H)
    y = torch.zeros(yh.shape[1]*hop+win).to(yh)
    for j in range(yh.shape[1]):
        y[j*hop:j*hop+win] += yh[:, j]
    return y

def load_audio(filename, sr=44100, mono=True):
    """
    Load an audio waveform from a file.  Try to use ffmpeg
    to convert it to a .wav file so scipy's fast wavfile loader
    can work.  Otherwise, fall back to the slower librosa

    Parameters
    ----------
    filename: string
        Path to audio file to load
    sr: int
        Sample rate to use
    mono: bool
        If true, use mono.  Otherwise, use stereo
    
    Returns
    -------
    y: ndarray(n_channels, N)
        Audio samples
    sr: int
        The sample rate that was actually used
    """
    # First, try a faster version of loading audio
    from scipy.io import wavfile
    import subprocess
    import os
    FFMPEG_BINARY = "ffmpeg"
    # Append a random int to the file in the rare use case that two processes
    # are running at the same time and loading the same file (rare case)
    if filename[-3:].lower() != "wav":
        wavfilename = "{}_{}.wav".format(filename, np.random.randint(999999999))
        if os.path.exists(wavfilename):
            os.remove(wavfilename)
        ac = "1"
        if not mono:
            ac = "2"
        subprocess.call([FFMPEG_BINARY, "-i", filename, "-ar", "%i"%sr, "-ac", ac, wavfilename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        _, y = wavfile.read(wavfilename)
        os.remove(wavfilename)
    else:
        _, y = wavfile.read(filename)
    y = y.T
    y = y/2.0**15
    if mono and len(y.shape) > 1 and y.shape[0] > 1:
        y = y[0, :]
    return y, sr

def load_corpus(path, sr, stereo, dc_normalize=True, amp_normalize=True, shift_min=0, shift_max=0):
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
    dc_normalize: bool
        If True, do a DC-offset normalization on each clip
    amp_normalize: bool
        If True (default), normalize the audio sample range to be in [-1, 1]
    shift_min: int
        Lowest halfstep by which to shift corpus
    shift_max: int
        Highest halfstep by which to shift corpus
    
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
        files = glob.glob(path + os.path.sep + "**", recursive=True)
        files = [f for f in files if os.path.isfile(f)]
    N = 0
    pitch_shift = librosa.effects.pitch_shift
    try:
        import pyrubberband
        pitch_shift = pyrubberband.pyrb.pitch_shift
        print("Rubberband is available for pitch shifting")
    except:
        rg = list(range(shift_min, shift_max+1))
        if len(rg) > 1 or rg[0] != 0:
            print("Warning: pyrubberband not found.  Using simpler phase vocoder for pitch shifting")
            
    for f in sorted(files):
        try:
            try:
                x, sr = librosa.load(f, sr=sr, mono=not stereo)
            except:
                x, sr = load_audio(f, sr=sr, mono=not stereo)
            x_orig = x
            for p in range(shift_min, shift_max+1):
                if p != 0:
                    print("Pitch shifting", f, "by", p, "halfsteps...")
                    xs = pitch_shift(x_orig, sr=sr, n_steps=p)
                    x = np.concatenate((x, xs), axis=1)
            if stereo and len(x.shape) == 1:
                x = np.array([x, x])
            if stereo:
                N += x.shape[1]
                if dc_normalize:
                    x -= np.mean(x, axis=1, keepdims=True)
            else:
                N += x.size
                if dc_normalize:
                    x -= np.mean(x)
            if amp_normalize:
                norm = np.max(np.abs(x))
                if norm > 0:
                    x = x/norm
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
