import numpy as np
import torch

def get_mel_filterbank(sr, win, mel_bands=40, min_freq=0.0, max_freq=8000):
    """
    Return a mel-spaced triangular filterbank

    Parameters
    ----------
    sr: int
        Audio sample rate
    win: int
        Window length
    mel_bands: int
        Number of bands to use
    min_freq: float 
        Minimum frequency for Mel filterbank
    max_freq: float
        Maximum frequency for Mel filterbank

    Returns
    -------
    ndarray(mel_bands, win//2+1)
        Mel filterbank, with each filter per row
    """
    melbounds = np.array([min_freq, max_freq])
    melbounds = 1125*np.log(1 + melbounds/700.0)
    mel = np.linspace(melbounds[0], melbounds[1], mel_bands+2)
    binfreqs = 700*(np.exp(mel/1125.0) - 1)
    binbins = np.floor(((win-1)/float(sr))*binfreqs) #Floor to the nearest bin
    binbins = np.array(binbins, dtype=np.int64)
    # Create mel triangular filterbank
    melfbank = np.zeros((mel_bands, win//2+1))
    for i in range(1, mel_bands+1):
        thisbin = binbins[i]
        lbin = binbins[i-1]
        rbin = thisbin + (thisbin - lbin)
        rbin = binbins[i+1]
        melfbank[i-1, lbin:thisbin+1] = np.linspace(0, 1, 1 + (thisbin - lbin))
        melfbank[i-1, thisbin:rbin+1] = np.linspace(1, 0, 1 + (rbin - thisbin))
    melfbank = melfbank/np.sum(melfbank, 1)[:, None]
    return melfbank

def get_dct_basis(N, n_dct=20):
    """
    Return a DCT Type-III basis

    Parameters
    ----------
    N: int
        Number of samples in signal
    n_dct: int
        Number of DCT basis elements

    Returns
    -------
    ndarray(n_dct, N)
        A matrix of the DCT basis
    """
    ts = np.arange(1, 2*N, 2)*np.pi/(2.0*N)
    fs = np.arange(1, n_dct)
    B = np.zeros((n_dct, N))
    B[1::, :] = np.cos(fs[:, None]*ts[None, :])*np.sqrt(2.0/N)
    B[0, :] = 1.0/np.sqrt(N)
    return B

class AudioFeatureComputer:
    def __init__(self, win=2048, sr=44100, min_freq=50, max_freq=8000, use_stft=True, mel_bands=40, use_mel=False, use_chroma=False, device="cpu"):
        """
        Parameters
        ----------
        win: int
            Window length
        sr: int
            Audio sample rate
        min_freq: float
            Minimum frequency of spectrogram, if using direct spectrogram features
        max_freq: float
            Maximum frequency of spectrogram, if using direct spectrogram features
        use_stft: bool
            If true, use straight up STFT bins
        mel_bands: int
            Number of bands to use if using mel-spaced STFT
        use_mel: bool
            If True, use mel-spaced STFT
        use_chroma: bool
            If True, use chroma features
        device: str
            Torch device on which to put features before returning
        """
        self.win = win
        self.sr = sr
        self.kmin = max(0, int(win*min_freq/sr)+1)
        self.kmax = min(int(win*max_freq/sr)+1, win//2)
        self.use_stft = use_stft
        self.use_mel = use_mel
        self.use_chroma = use_chroma
        self.device = device

        if self.use_mel:
            self.M = get_mel_filterbank(sr, win, mel_bands, min_freq, max_freq)
    
    def __call__(self, x):
        """
        Parameters
        ----------
        x: ndarray(win) or ndarray(win, n_frames)
            Pre-windowed audio frames
        """
        if len(x.shape) == 1:
            x = x[:, None]
        S = np.abs(np.fft.rfft(x, axis=0))
        components = []
        if self.use_stft:
            # Ordinary STFT
            components.append(S[self.kmin:self.kmax, :])
        if self.use_mel:
            components.append(self.M.dot(S))
        res = np.concatenate(tuple(components), axis=0)
        if x.shape == 0:
            res = res[:, 0]
        return torch.from_numpy(np.array(res, dtype=np.float32)).to(self.device)