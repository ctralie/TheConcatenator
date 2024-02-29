import numpy as np
import torch

def get_mel_filterbank(sr, win, n_bands=40, fmin=0.0, fmax=8000):
    """
    Return a mel-spaced triangular filterbank

    Parameters
    ----------
    sr: int
        Audio sample rate
    win: int
        Window length
    n_bands: int
        Number of bands to use
    fmin: float 
        Minimum frequency for Mel filterbank
    fmax: float
        Maximum frequency for Mel filterbank

    Returns
    -------
    ndarray(n_bands, win//2+1)
        Mel filterbank, with each filter per row
    """
    melbounds = np.array([fmin, fmax])
    melbounds = 1125*np.log(1 + melbounds/700.0)
    mel = np.linspace(melbounds[0], melbounds[1], n_bands+2)
    binfreqs = 700*(np.exp(mel/1125.0) - 1)
    binbins = np.floor(((win-1)/float(sr))*binfreqs) #Floor to the nearest bin
    binbins = np.array(binbins, dtype=np.int64)

    #Step 2: Create mel triangular filterbank
    melfbank = np.zeros((n_bands, win//2+1))
    for i in range(1, n_bands+1):
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
    def __init__(self, win=2048, sr=44100, min_freq=0, max_freq=0, use_mfcc=False, use_chroma=False, device="cpu"):
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
        use_mfcc: bool
            If True, use MFCC features
        use_chroma: bool
            If True, use chroma features
        device: str
            Torch device on which to put features before returning
        """
        self.win = win
        self.sr = sr
        self.kmin = max(0, int(win*min_freq/sr)+1)
        self.kmax = min(int(win*max_freq/sr)+1, win//2)
        self.use_mfcc = use_mfcc
        self.use_chroma = use_chroma
        self.device = device

        if self.use_mfcc:
            self.n_mel_bands = 40
            self.n_dct = 20
            self.lifter_coeffs = np.arange(self.n_dct)**0.6
            self.lifter_coeffs[0] = 1
            self.M = get_mel_filterbank(sr, win, self.n_mel_bands)
            self.B = get_dct_basis(self.n_mel_bands, self.n_dct)

    def get_feature(self, x):
        """
        Parameters
        ----------
        x: ndarray(win)

        """
        f = np.abs(np.fft.rfft(x))
        x = np.array([])
        if self.kmax-self.kmin > 0:
            # Ordinary STFT
            x = np.concatenate((x, f[self.kmin:self.kmax]))
        if self.use_mfcc:
            xmel = self.M.dot(f)
            xmel = 10*np.log10(np.maximum(1e-10, xmel))
            xmfcc = self.lifter_coeffs*(self.B.dot(xmel))
            x = np.concatenate((x, xmfcc))
        return torch.from_numpy(np.array(x, dtype=np.float32)).to(self.device)

    def __call__(self, x):
        """
        Parameters
        ----------
        x: ndarray(win) or ndarray(win, n_frames)
            Pre-windowed audio frames
        """
        W = None
        if len(x.shape) == 2:
            ## Compute features of each window
            W = [self.get_feature(x[:, j]).unsqueeze(-1) for j in range(x.shape[1])]
            W = torch.concatenate(tuple(W), dim=1)
        else:
            W = self.get_feature(x)
        return W