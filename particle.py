import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from probutils import stochastic_universal_sample, do_KL, count_top_activations, get_activations_diff, get_repeated_activation_itervals, get_diag_lengths
from observer import Observer
from propagator import Propagator
from audioutils import get_windowed, hann_window
import pyaudio
import struct
import time

class ParticleFilter:
    def __init__(self, ycorpus, win, sr, min_freq, max_freq, p, pfinal, pd, temperature, L, P, gamma=0, r=3, neff_thresh=0, use_gpu=True, use_mic=False):
        """
        ycorpus: ndarray(n_channels, n_samples)
            Stereo audio samples for the corpus
        win: int
            Window length for each STFT window.  For simplicity, assume
            that hop is 1/2 of this
        sr: int
            Audio sample rate
        min_freq: float
            Minimum frequency to use (in hz)
        max_freq: float
            Maximum frequency to use (in hz)
        p: int
            Sparsity parameter for particles
        pfinal: int
            Sparsity parameter for final activations
        pd: float
            State transition probability
        temperature: float
            Amount to focus on matching observations
        L: int
            Number of iterations for NMF observation probabilities
        P: int
            Number of particles
        gamma: float
            Cosine similarity cutoff
        r: int
            Repeated activations cutoff
        neff_thresh: float
            Number of effective particles below which to resample
        use_gpu: bool
            Whether to use the gpu for the observation probabilities
        use_mic: bool
            If true, use the microphone
        """
        self.win_samples = hann_window(win)
        self.pfinal = pfinal
        self.pd = pd
        self.temperature = temperature
        self.L = L
        self.gamma = gamma
        self.r = r
        self.neff_thresh = neff_thresh
        self.use_gpu = use_gpu
        self.use_mic = use_mic
        self.win = win
        self.sr = sr
        self.kmin = max(0, int(win*min_freq/sr)+1)
        self.kmax = min(int(win*max_freq/sr)+1, win//2)
        print(self.kmin, self.kmax)
        hop = win//2

        self.neff = [] # Number of effective particles over time
        self.wsmax = [] # Max weights over time
        self.ws = [] # Weights over time
        self.topcounts = [] 
        self.frame_times = [] # Keep track of time to process each frame
        self.chosen_idxs = [] # Keep track of chosen indices
        self.H = [] # Activations of chosen indices

        ## Step 1: Compute spectrogram for corpus
        n_channels = ycorpus.shape[0]
        self.WSound = [get_windowed(ycorpus[i, :], hop, win, hann_window) for i in range(n_channels)]
        Ws = [np.abs(np.fft.rfft(W, axis=0)[self.kmin:self.kmax, :]) for W in self.WSound]
        WCorpus = np.concatenate(tuple(Ws), axis=0)
        self.WCorpus = WCorpus

        ## Step 2: Setup KDTree for proposal indices if requested
        self.proposal_tree = None
        if gamma > 0:
            WMag = np.sqrt(np.sum(WCorpus.T**2, axis=1))
            WMag[WMag == 0] = 1
            WNorm = WCorpus.T/WMag[:, None] # Vector normalized version for KDTree
            self.proposal_tree = KDTree(WNorm)
            self.proposal_d = 2*(1-gamma)**0.5 # KDTree distance corresponding to gamma cosine similarity
        
        ## Step 3: Setup observer and propagator
        N = WCorpus.shape[1]
        WDenom = np.sum(WCorpus, axis=0)
        WDenom[WDenom == 0] = 1
        self.observer = Observer(p, WCorpus/WDenom, L)
        self.propagator = Propagator(N, pd)
        self.states = np.random.randint(N, size=(P, p)) # Particles
        self.ws = np.ones(P)/P # Particle weights
        self.all_ws = []
        self.fit = 0 # KL fit

        ## Step 4: Setup a circular buffer that receives hop samples at a time
        self.buf_in  = np.zeros((n_channels, win))
        # Setup an output buffer that doubles in size like an arraylist
        self.buf_out = np.zeros((n_channels, sr*60*10))

        ## Step 5: If we're using the mic, set that up
        if use_mic:
            audio = pyaudio.PyAudio()
            ## TODO: Finish this
            

    def get_H(self):
        """
        Convert chosen_idxs and H into a numpy array with 
        activations in the proper indices

        Returns
        -------
        H: ndarray(N, T)
            Activations of the corpus over time
        """
        from scipy import sparse
        N = self.WCorpus.shape[1]
        p = self.states.shape[1]
        T = len(self.H)
        vals = np.array(self.H).flatten()
        rows = np.array(self.chosen_idxs, dtype=int).flatten()
        cols = np.array(np.ones((1, p))*np.arange(T)[:, None], dtype=int).flatten()
        H = sparse.coo_matrix((vals, (rows, cols)), shape=(N, T))
        return H.toarray()
    
    def get_generated_audio(self):
        """
        Return the audio that's been generated

        Returns
        -------
        ndarray(n_samples, 2)
            Generated audio
        """
        T = len(self.chosen_idxs)
        hop = self.win//2
        ret = self.buf_out[:, 0:T*hop]
        ret /= np.max(np.abs(ret))
        return ret.T

    def audio_in(self, s, frame_count=None, time_info=None, status=None):
        """
        Incorporate win//2 audio samples, either directly from memory or from
        the microphone

        Parameters
        ----------
        s: byte string or ndarray
            Byte string of audio samples if using mic, or ndarray(win//2)
            samples if not using mic
        frame_count: int
            If using mic, it should be win//2 samples
        """
        tic = time.time()
        if self.use_mic:
            fmt = "<"+"h"*frame_count
            x = struct.unpack(fmt, s)
            x = np.array(x, dtype=float)/32768
        else:
            x = s
        hop = self.win//2
        self.buf_in[:, 0:hop] = self.buf_in[:, hop:]
        self.buf_in[:, hop:] = x
        self.process_window(self.buf_in)
        # Record elapsed time
        elapsed = time.time()-tic
        self.frame_times.append(elapsed)
    
    def audio_out(self, x):
        """
        Incorporate a new window into the output audio buffer

        x: ndarray(2, win)
            Windowed stereo audio
        """
        win = self.win
        hop = win//2
        T = len(self.chosen_idxs)
        N = self.buf_out.shape[1]
        if N < (T+1)*hop:
            # Double size
            new_out = np.zeros((2, N*2))
            new_out[:, 0:N] = self.buf_out
            self.buf_out = new_out
        idx = hop*(T-1) # Start index
        self.buf_out[:, idx:idx+win] += x
        # Ready to output hop more samples
        ## TODO: Fill this in
        
    def process_audio_offline(self, ytarget):
        """
        Process audio audio offline, frame by frame

        Parameters
        ----------
        ytarget: ndarray(n_channels, T)
            Audio samples to process

        Returns
        -------
        ndarray(n_samples, n_channels)
            Generated audio
        """
        hop = self.win//2
        for i in range(0, ytarget.shape[1]//hop):
            self.audio_in(ytarget[:, i*hop:(i+1)*hop])
        return self.get_generated_audio()
    
    def process_window(self, x):
        """
        x: ndarray(n_channels, win)
            Window to process
        """
        if len(self.H)%10 == 0:
            print(".", end="", flush=True)
        
        p = self.states.shape[1]
        ## Step 1: Do STFT of this window and sample from proposal distribution
        Vs = [np.abs(np.fft.rfft(self.win_samples*x[i, :], axis=0)[self.kmin:self.kmax]) for i in range(x.shape[0])]
        Vt = np.concatenate(tuple(Vs))[:, None]
        self.propagator.propagate_numba(self.states)

        ## Step 2: Apply the observation probability updates
        dots = []
        if self.use_gpu:
            dots = self.observer.observe(self.states, Vt)
        else:
            dots = self.observer.observe_cpu(self.states, Vt)
        obs_prob = np.exp(dots*self.temperature/np.max(dots))
        obs_prob /= np.sum(obs_prob)
        self.ws *= obs_prob

        ## Step 3: Figure out the activations for this timestep
        ## by aggregating multiple particles near the top
        self.wsmax.append(np.max(self.ws))
        N = self.WCorpus.shape[1]
        probs = np.zeros(N)
        max_particles = np.argpartition(-self.ws, 2*p)[0:2*p]
        for state, w in zip(self.states[max_particles], self.ws[max_particles]):
            probs[state] += w
        # Promote states that follow the last state that was chosen
        if len(self.chosen_idxs) > 0:
            last_state = self.chosen_idxs[-1] + 1
            probs[last_state[last_state < N]] *= 5
        # Zero out last ones to prevent repeated activations
        for dc in range(1, min(self.r, len(self.chosen_idxs))+1):
            probs[self.chosen_idxs[-dc]] = 0
        top_idxs = np.argpartition(-probs, self.pfinal)[0:self.pfinal]
        
        self.chosen_idxs.append(top_idxs)
        h = do_KL(self.WCorpus[:, top_idxs], Vt[:, 0], self.L)
        self.H.append(h)
        
        ## Step 4: Resample particles if effective number is too low
        self.ws /= np.sum(self.ws)
        self.all_ws.append(self.ws)
        self.neff.append(1/np.sum(self.ws**2))
        if self.neff[-1] < self.neff_thresh:
            choices, _ = stochastic_universal_sample(self.ws, len(self.ws))
            self.states = self.states[choices, :]
            self.ws = np.ones(self.ws.size)/self.ws.size
        
        ## Step 5: Create and output audio samples for this window
        y = np.zeros_like(x)
        for i in range(x.shape[0]):
            y[i, :] = self.WSound[i][:, top_idxs].dot(h)
        self.audio_out(y)

        ## Step 6: Accumulate KL term for fit
        WH = self.WCorpus[:, top_idxs].dot(h)
        Vt = Vt.flatten()
        self.fit += np.sum(Vt*np.log(Vt/WH) - Vt + WH)

    def plot_statistics(self):
        """
        Plot statistics about the activations that were chosen
        """
        p = self.states.shape[1]
        H = self.get_H()

        active_diffs = get_activations_diff(H, p)
        repeated_intervals = get_repeated_activation_itervals(H, p)
        
        plt.subplot(231)
        plt.plot(active_diffs)
        plt.legend(["Particle Filter: Mean {:.3f}".format(np.mean(active_diffs))])
        plt.title("Activation Changes over Time, Temperature {}".format(self.temperature))
        plt.xlabel("Timestep")

        plt.subplot(232)
        plt.hist(active_diffs, bins=np.arange(30))
        plt.title("Activation Changes Histogram")
        plt.xlabel("Number of Activations Changed")
        plt.ylabel("Counts")
        plt.legend(["Ground Truth", "Particle Filter"])

        plt.subplot(233)
        plt.hist(repeated_intervals, bins=np.arange(30))
        plt.title("Repeated Activations Histogram, r={}".format(self.r))
        plt.xlabel("Repeated Activation Distance")
        plt.ylabel("Counts")
        plt.legend(["Ground Truth", "Particle Filter"])

        plt.subplot(234)
        plt.plot(self.wsmax)
        plt.title("Max Probability, Overall Fit: {:.3f}".format(self.fit))
        plt.xlabel("Timestep")

        plt.subplot(235)
        plt.plot(self.neff)
        plt.xlabel("Timestep")
        plt.title("Number of Effective Particles (Median {:.2f})".format(np.median(self.neff)))

        plt.subplot(236)
        diags = get_diag_lengths(H, p)
        plt.hist(diags, bins=np.arange(30))
        plt.legend(["Particle Filter Mean: {:.3f} ($p_d$={})".format(np.mean(diags), self.pd)])
        plt.xlabel("Diagonal Length")
        plt.ylabel("Counts")
        plt.title("Diagonal Lengths")