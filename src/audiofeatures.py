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

def get_yin(F):
    """
    Compute the normalized yin on windowed audio

    Parameters
    ----------
    F: ndarray(win_length, n_win)
        Windowed audio
    
    Returns
    -------
    ndarray(win_length, n_win)
        Normalized yin
    """
    win_length = F.shape[0]
    hop = win_length//2

    ## Step 1: Do autocorrelation
    a = np.fft.rfft(F, axis=0)
    acf = np.fft.irfft(a*np.conj(a), axis=0)[0:hop, :]
    ## Step 2: Compute windowed energy
    energy = np.cumsum(F**2, axis=0)
    energy = energy[hop::, :] - energy[0:-hop, :]

    ## Step 3: Yin and normalized yin
    yin = energy[0, :] + energy - 2*acf
    yin -= np.min(yin, axis=0, keepdims=True)
    denom = np.cumsum(yin[1::, :], axis=0)
    denom[denom<1e-6] = 1
    nyin = np.ones(yin.shape)
    nyin[1::, :] = yin[1::, :]*np.arange(1, yin.shape[0])[:, None]/denom

    ret = nyin[0:hop//2, :]
    ret[ret < 0] = 0
    return ret

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
    melfbank = np.zeros((mel_bands, win//2+1), dtype=np.float32)
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
    B = np.zeros((n_dct, N), dtype=np.float32)
    B[1::, :] = np.cos(fs[:, None]*ts[None, :])*np.sqrt(2.0/N)
    B[0, :] = 1.0/np.sqrt(N)
    return B

class AudioFeatureComputer:
    def __init__(self, win=2048, sr=44100, min_freq=50, max_freq=8000, use_stft=True, mel_bands=40, use_mel=False, use_superflux=False, device="cpu"):
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
        use_superflux: bool
            If True, using superflux for audio novelty
        device: str
            Torch device on which to put features before returning
        """
        self.win = win
        self.sr = sr
        self.kmin = max(0, int(win*min_freq/sr)+1)
        self.kmax = min(int(win*max_freq/sr)+1, win//2)
        self.use_stft = use_stft
        self.use_mel = use_mel
        self.use_superflux = use_superflux
        self.device = device
        
        if use_stft and use_mel:
            print("Warning: Using both STFT and Mel-spacing simultaneously.  Did you mean this?")

        if self.use_mel:
            self.M = get_mel_filterbank(sr, win, mel_bands, min_freq, max_freq)

    def get_spectral_features(self, x, shift=0):
        """
        Compute the spectral features, which are used in KL fit

        Parameters
        ----------
        x: ndarray(win) or ndarray(win, n_frames)
            Pre-windowed audio frames
        shift: float
            Amount by which to shift this feature
        """
        if len(x.shape) == 1:
            x = x[:, None]
        if shift != 0:
            from librosa.effects import pitch_shift
            for k in range(x.shape[1]):
                x[:, k] = pitch_shift(x[:, k], sr=self.sr, n_steps=shift)
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
        res = np.array(res, dtype=np.float32)
        if self.device != "np":
            from torch import from_numpy
            res = from_numpy(res).to(self.device)
        return res