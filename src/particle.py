import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from probutils import stochastic_universal_sample, do_KL_torch, count_top_activations, get_activations_diff, get_repeated_activation_itervals, get_diag_lengths
from observer import Observer
from propagator import Propagator
from audioutils import get_windowed, hann_window
from audiofeatures import AudioFeatureComputer
import pyaudio
import struct
import time
from threading import Lock
from tkinter import Tk, ttk

class ParticleFilter:
    def __init__(self, ycorpus, feature_params, particle_params, device="cpu", use_mic=False):
        """
        ycorpus: ndarray(n_channels, n_samples)
            Stereo audio samples for the corpus
        feature_params: {
            win: int
                Window length for each STFT window.  For simplicity, assume
                that hop is 1/2 of this
            sr: int
                Audio sample rate
            min_freq: float
                Minimum frequency to use (in hz)
            max_freq: float
                Maximum frequency to use (in hz)
            use_stft: bool
                If true, use straight up STFT bins
            mel_bands: int
                Number of bands to use if using mel-spaced STFT
            use_mel: bool
                If True, use mel-spaced STFT
            use_chroma: bool
                If True, use chroma features
            use_zcs: bool
                If True, use zero crossings
        }
        particle_params: {
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
        }
        device: string
            Device string for torch
        use_mic: bool
            If true, use the microphone
        """
        print("Setting up particle filter...")
        tic = time.time()
        win = feature_params["win"]
        sr = feature_params["sr"]
        self.win_samples = np.array(hann_window(win), dtype=np.float32)
        self.p = particle_params["p"]
        self.P = particle_params["P"]
        self.pfinal = particle_params["pfinal"]
        self.pd = particle_params["pd"]
        self.temperature = particle_params["temperature"]
        self.L = particle_params["L"]
        self.gamma = particle_params["gamma"]
        self.r = particle_params["r"]
        self.neff_thresh = particle_params["neff_thresh"]
        self.device = device
        self.use_mic = use_mic
        self.win = win
        self.sr = sr
        hop = win//2
        self.hop = hop

        self.neff = [] # Number of effective particles over time
        self.wsmax = [] # Max weights over time
        self.ws = [] # Weights over time
        self.topcounts = [] 
        self.frame_times = [] # Keep track of time to process each frame
        self.chosen_idxs = [] # Keep track of chosen indices
        self.H = [] # Activations of chosen indices

        ## Step 1: Compute features for corpus
        feature_params["device"] = device
        self.feature_computer = AudioFeatureComputer(**feature_params)
        n_channels = ycorpus.shape[0]
        self.n_channels = n_channels
        self.WSound = [get_windowed(ycorpus[i, :], hop, win, hann_window) for i in range(n_channels)]
        WCorpus = torch.concatenate(tuple([self.feature_computer(W) for W in self.WSound]), axis=0)
        self.WCorpus = WCorpus

        ## Step 2: Setup KDTree for proposal indices if requested
        """
        ## TODO: Finish this (useful for huuuuge corpuses)
        self.proposal_tree = None
        if gamma > 0:
            WMag = np.sqrt(np.sum(WCorpus.T**2, axis=1))
            WMag[WMag == 0] = 1
            WNorm = WCorpus.T/WMag[:, None] # Vector normalized version for KDTree
            self.proposal_tree = KDTree(WNorm)
            self.proposal_d = 2*(1-gamma)**0.5 # KDTree distance corresponding to gamma cosine similarity
        """
        
        ## Step 3: Setup observer and propagator
        N = WCorpus.shape[1]
        WDenom = torch.sum(WCorpus, dim=0, keepdims=True)
        WDenom[WDenom == 0] = 1
        self.observer = Observer(self.p, WCorpus/WDenom, self.L)
        self.propagator = Propagator(N, self.pd, device)
        self.states = torch.randint(N, size=(self.P, self.p), dtype=torch.int32).to(device) # Particles
        self.ws = np.array(np.ones(self.P)/self.P, dtype=np.float32)
        self.ws = torch.from_numpy(self.ws).to(device) # Particle weights
        self.all_ws = []
        self.fit = 0 # KL fit

        ## Step 3b: Run observer, propagator, and resampler on random data to precompile kernels
        ## so that the first step doesn't lag
        states_dummy = torch.randint(N, size=(self.P, self.p), dtype=torch.int32).to(device)
        self.observer.observe(states_dummy, torch.rand(WCorpus.shape[0], 1, dtype=torch.float32).to(device))
        self.propagator.propagate(states_dummy)
        stochastic_universal_sample(np.random.rand(self.P), self.P)

        ## Step 4: Setup audio buffers
        # Setup a circular buffer that receives in hop samples at a time
        self.buf_in  = np.zeros((n_channels, win), dtype=np.float32)
        # Setup an output buffer that doubles in size like an arraylist
        self.buf_out = np.zeros((n_channels, sr*60*10), dtype=np.float32)

        print("Finished setting up particle filter: Elapsed Time {:.3f} seconds".format(time.time()-tic))

        ## Step 5: If we're using the mic, set that up
        if use_mic:
            self.recorded_audio = []
            self.mutex = Lock()
            self.processing_frame = False
            self.audio = pyaudio.PyAudio()
            self.recording_started = False
            self.recording_finished = False
            self.tk_root = Tk()
            f = ttk.Frame(self.tk_root, padding=10)
            f.grid()
            ttk.Label(f, text="The Concatenator Real Time!").grid(column=0, row=0)
            self.record_button = ttk.Button(f, text="Start Recording", command=self.start_audio_recording)
            self.record_button.grid(column=0, row=1)
            self.tk_root.mainloop()
            

    def start_audio_recording(self):
        self.stream = self.audio.open(format=pyaudio.paFloat32, 
                            frames_per_buffer=self.hop, 
                            channels=self.n_channels, 
                            rate=self.sr, 
                            output=True, 
                            input=True, 
                            stream_callback=self.audio_in)
        self.record_button.configure(text="Stop Recording")
        self.record_button.configure(command=self.stop_audio_recording)
        self.recording_started = True
        self.stream.start_stream()
    
    def stop_audio_recording(self):
        self.stream.close()
        self.audio.terminate()
        self.tk_root.destroy()
        self.recording_finished = True
    
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
        vals = np.array([h.cpu().numpy() for h in self.H]).flatten()
        rows = np.array([c.cpu().numpy() for c in self.chosen_idxs], dtype=int).flatten()
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
        if ret.size > 0:
            ret /= np.max(np.abs(ret))
        return ret.T
    
    def get_recorded_audio(self):
        """
        Returns the audio that's been recorded if we've
        done a recording session
        """
        assert(self.use_mic)
        return np.concatenate(tuple(self.recorded_audio), axis=1).T

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
            return_early = False
            with self.mutex:
                if self.processing_frame:
                    # We're already in the middle of processing a frame,
                    # so pass the audio through
                    return_early = True
                else:
                    self.processing_frame = True
            if return_early:
                print("Returning early", flush=True)
                return s, pyaudio.paContinue
            nc = self.n_channels
            fmt = "<"+"f"*(self.n_channels*self.win//2)
            x = np.array(struct.unpack(fmt, s), dtype=np.float32)
            x = np.reshape(x, (x.size//nc, nc)).T
            self.recorded_audio.append(x)
        else:
            x = s
        hop = self.win//2
        self.buf_in[:, 0:hop] = self.buf_in[:, hop:]
        self.buf_in[:, hop:] = x
        self.process_window(self.buf_in)
        # Record elapsed time
        elapsed = time.time()-tic
        self.frame_times.append(elapsed)
        # Output the audio that's ready
        T = len(self.chosen_idxs)
        idx = hop*(T-1) # Start index
        ret = (self.buf_out[:, idx:idx+hop].T).flatten()
        if self.use_mic:
            with self.mutex:
                self.processing_frame = False
        return struct.pack("<"+"f"*ret.size, *ret), pyaudio.paContinue
    
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
            new_out = np.zeros((2, N*2), dtype=np.float32)
            new_out[:, 0:N] = self.buf_out
            self.buf_out = new_out
        idx = hop*(T-1) # Start index
        self.buf_out[:, idx:idx+win] += x
        # Now ready to output hop more samples
        
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
        Vs = [self.feature_computer(self.win_samples*x[i, :]) for i in range(x.shape[0])]
        Vt = torch.concatenate(tuple(Vs))
        Vt = Vt.view(Vt.numel(), 1)
        self.propagator.propagate(self.states)

        ## Step 2: Apply the observation probability updates
        dots = self.observer.observe(self.states, Vt)
        dots[dots < 1e-7] = 1e-7
        obs_prob = self.temperature*torch.log(dots)
        obs_prob -= torch.max(obs_prob)
        obs_prob = torch.exp(obs_prob)
        obs_prob /= torch.sum(obs_prob)
        self.ws *= obs_prob

        ## Step 3: Figure out the activations for this timestep
        ## by aggregating multiple particles near the top
        self.wsmax.append(torch.max(self.ws).item())
        N = self.WCorpus.shape[1]
        probs = torch.zeros(N).to(self.ws)
        max_particles = torch.topk(self.ws, 2*p, largest=True)[1]
        for state, w in zip(self.states[max_particles], self.ws[max_particles]):
            probs[state] += w
        # Promote states that follow the last state that was chosen
        if len(self.chosen_idxs) > 0:
            last_state = self.chosen_idxs[-1] + 1
            probs[last_state[last_state < N]] *= 5
        # Zero out last ones to prevent repeated activations
        for dc in range(1, min(self.r, len(self.chosen_idxs))+1):
            probs[self.chosen_idxs[-dc]] = 0
        top_idxs = torch.topk(probs, self.pfinal, largest=True)[1]
        self.chosen_idxs.append(top_idxs)

        h = do_KL_torch(self.WCorpus[:, top_idxs], Vt[:, 0], self.L)
        self.H.append(h)
        
        ## Step 4: Resample particles if effective number is too low
        self.ws /= torch.sum(self.ws)
        self.all_ws.append(self.ws.cpu().numpy())
        self.neff.append((1/torch.sum(self.ws**2)).item())
        if self.neff[-1] < self.neff_thresh:
            ## TODO: torch-ify stochastic universal sample
            choices, _ = stochastic_universal_sample(self.all_ws[-1], len(self.ws))
            choices = torch.from_numpy(np.array(choices, dtype=int)).to(self.device)
            self.states = self.states[choices, :]
            self.ws = torch.ones(self.ws.shape).to(self.ws)/self.ws.numel()
        
        ## Step 5: Create and output audio samples for this window
        y = np.zeros_like(x)
        for i in range(x.shape[0]):
            y[i, :] = self.WSound[i][:, top_idxs.cpu().numpy()].dot(h.cpu().numpy())
        self.audio_out(y)

        ## Step 6: Accumulate KL term for fit
        WH = torch.matmul(self.WCorpus[:, top_idxs], h)
        Vt = Vt.flatten()
        self.fit += (torch.sum(Vt*torch.log(Vt/WH) - Vt + WH)).item()

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