import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from probutils import stochastic_universal_sample, do_KL_torch, count_top_activations, get_activations_diff, get_repeated_activation_itervals, get_diag_lengths
from observer import Observer
from propagator import Propagator
from audioutils import get_windowed, hann_window, tri_window
from audiofeatures import AudioFeatureComputer
import pyaudio
import struct
import time
from threading import Lock
from tkinter import Tk, ttk

CORPUS_DB_CUTOFF = -50

class ParticleFilter:
    def reset_state(self):
        self.neff = [] # Number of effective particles over time
        self.wsmax = [] # Max weights over time
        self.ws = [] # Weights over time
        self.topcounts = [] 
        self.frame_times = [] # Keep track of time to process each frame
        self.chosen_idxs = [] # Keep track of chosen indices
        self.H = [] # Activations of chosen indices

        self.states = torch.randint(self.N, size=(self.P, self.p), dtype=torch.int32).to(self.device) # Particles
        self.ws = np.array(np.ones(self.P)/self.P, dtype=np.float32)
        self.ws = torch.from_numpy(self.ws).to(self.device) # Particle weights
        self.all_ws = []
        self.fit = 0 # KL fit
        self.num_resample = 0
        
        # Setup a circular buffer that receives in hop samples at a time
        self.buf_in  = np.zeros((self.n_channels, self.win), dtype=np.float32)
        # Setup an output buffer that doubles in size like an arraylist
        self.buf_out = np.zeros((self.n_channels, self.sr*60*10), dtype=np.float32)

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
            proposal_k: float
                Number of nearest neighbors to use in proposal distribution
                (if 0, don't use proposal distribution)
            r: int
                Repeated activations cutoff
            neff_thresh: float
                Number of effective particles below which to resample
            alpha: float
                L2 penalty for weights
            use_top_particle: bool
                If True, only take activations from the top particle at each step.
                If False, aggregate 
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
        self.p = particle_params["p"]
        self.P = particle_params["P"]
        self.pfinal = particle_params["pfinal"]
        self.pd = particle_params["pd"]
        self.temperature = particle_params["temperature"]
        self.L = particle_params["L"]
        self.proposal_k = particle_params["proposal_k"]
        self.r = particle_params["r"]
        self.neff_thresh = particle_params["neff_thresh"]
        self.alpha = particle_params["alpha"]
        self.use_top_particle = particle_params["use_top_particle"]
        self.device = device
        self.use_mic = use_mic
        self.win = win
        self.sr = sr
        hop = win//2
        self.hop = hop

        ## Step 1: Compute features for corpus
        feature_params["device"] = device
        self.feature_computer = AudioFeatureComputer(**feature_params)
        n_channels = ycorpus.shape[0]
        self.n_channels = n_channels
        self.WSound = []
        WPowers = [] # Store the maximum power over all channels
        for i in range(n_channels):
            WSi, WPi = get_windowed(ycorpus[i, :], hop, win, hann_window)
            self.WSound.append(WSi)
            if i == 0:
                WPowers = WPi
            else:
                WPowers = np.maximum(WPowers, WPi)
        # Corpus is analyzed with hann window
        self.win_samples = np.array(hann_window(win), dtype=np.float32)
        WCorpus = torch.concatenate(tuple([self.feature_computer(W) for W in self.WSound]), axis=0)
        self.WCorpus = WCorpus
        # Shrink elements that are too small
        self.WAlpha = self.alpha*np.array(WPowers <= CORPUS_DB_CUTOFF, dtype=np.float32)
        self.WAlpha = torch.from_numpy(self.WAlpha).to(self.device)
        self.loud_enough_idx_map = np.arange(WCorpus.shape[1])[WPowers > CORPUS_DB_CUTOFF]
        print("{:.3f}% of corpus is above loudness threshold".format(100*self.loud_enough_idx_map.size/WCorpus.shape[1]))

        ## Step 2: Setup KDTree for proposal indices if requested
        self.proposal_tree = None
        if self.proposal_k > 0:
            WCorpusNumpy = WCorpus.cpu().numpy()
            WMag = np.sqrt(np.sum(WCorpusNumpy.T**2, axis=1))
            WMag[WMag == 0] = 1
            WNorm = WCorpusNumpy.T/WMag[:, None] # Vector normalized version for KDTree
            self.proposal_tree = KDTree(WNorm[self.loud_enough_idx_map, :], leaf_size=30, metric='euclidean')
        
        ## Step 3: Setup observer and propagator
        N = WCorpus.shape[1]
        self.N = N
        self.observer = Observer(self.p, WCorpus, self.WAlpha, self.L, self.temperature)
        self.propagator = Propagator(N, self.pd, device)
        self.reset_state()

        print("Finished setting up particle filter: Elapsed Time {:.3f} seconds".format(time.time()-tic))

        ## Step 5: If we're using the mic, set that up
        if use_mic:
            self.setup_mic()
            
    def update_temperature(self, value):
        with self.temperature_mutex:
            self.temperature = float(value)
        self.temp_label.config(text="temperature ({:.1f})".format(self.temperature))
    
    def update_pd(self, value):
        self.pd = float(value)
        self.pd_label.config(text="pd ({:.5f})".format(self.pd))
        self.propagator.update_pd(self.pd)
    
    def update_pfinal(self, value):
        self.pfinal = int(float(value))
        self.pfinal_label.config(text="pfinal ({})".format(self.pfinal))
        self.pfinal_slider.config(value=self.pfinal)

    def setup_mic(self):
        hop = self.win//2
        self.recorded_audio = []
        self.frame_mutex = Lock()
        self.temperature_mutex = Lock()
        self.processing_frame = False
        self.audio = pyaudio.PyAudio()
        self.recording_started = False
        self.recording_finished = False

        ## Step 1: Setup menus
        self.tk_root = Tk()
        f = ttk.Frame(self.tk_root, padding=10)
        f.grid()
        row = 0
        ttk.Label(f, text="The Concatenator Real Time!").grid(column=0, row=row)
        row += 1
        self.record_button = ttk.Button(f, text="Start Recording", command=self.start_audio_recording)
        self.record_button.grid(column=0, row=row)
        row += 1
        # Temperature slider
        self.temp_label = ttk.Label(f, text="temperature")
        self.temp_label.grid(column=1, row=row)
        self.temp_slider = ttk.Scale(f, from_=0, to=max(50, self.temperature*1.5), length=400, value=self.temperature, orient="horizontal")
        ## TODO: This is a hacky way to get things to update on release!
        fn = lambda _: self.update_temperature(self.temp_slider.get())
        for i in [1, 2, 3]:
            self.temp_slider.bind("<ButtonRelease-{}>".format(i), fn) 
        self.temp_slider.grid(column=0, row=row)
        self.update_temperature(self.temperature)
        row += 1
        # pd slider
        self.pd_label = ttk.Label(f, text="pd")
        self.pd_label.grid(column=1, row=row)
        mx = 1-2/self.P # Leave enough room to jump to at least one particle
        self.pd_slider = ttk.Scale(f, from_=0.5, to=mx, length=400, value=self.pd, orient="horizontal")
        ## TODO: This is a hacky way to get things to update on release!
        fn = lambda _: self.update_pd(self.pd_slider.get())
        for i in [1, 2, 3]:
            self.pd_slider.bind("<ButtonRelease-{}>".format(i), fn) 
        self.pd_slider.grid(column=0, row=row)
        self.update_pd(self.pd)
        row += 1
        """
        # pfinal slider (TODO: Finish this)
        self.pfinal_label = ttk.Label(f, text="pfinal")
        self.pfinal_label.grid(column=1, row=row)
        self.pfinal_slider = ttk.Scale(f, from_=1, to=self.p, length=400, value=self.pfinal, orient="horizontal", command=self.update_pfinal)
        self.pfinal_slider.grid(column=0, row=row)
        self.update_pfinal(self.pfinal)
        """

        ## Step 2: Run one frame with dummy data to precompile all kernels
        ## Use high amplitude random noise to get it to jump around a lot
        bstr = np.array(np.random.rand(hop*2), dtype=np.float32)
        bstr = struct.pack("<"+"f"*hop*2, *bstr)
        for _ in range(20):
            self.audio_in(bstr)
        self.reset_state()
        self.recorded_audio = []
        self.buf_in *= 0
        self.buf_out *= 0

        ## Step 3: Start loop!
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
        T = len(self.H)
        vals = np.array([h.cpu().numpy() for h in self.H]).flatten()
        print("Min h: {:.3f}, Max h: {:.3f}".format(np.min(vals), np.max(vals)))
        rows = np.array(self.chosen_idxs, dtype=int).flatten()
        cols = np.array(np.ones((1, self.pfinal))*np.arange(T)[:, None], dtype=int).flatten()
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
            with self.frame_mutex:
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
            with self.frame_mutex:
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
        if len(ytarget.shape) == 1:
            ytarget = ytarget[None, :] # Mono audio
        hop = self.win//2
        for i in range(0, ytarget.shape[1]//hop):
            self.audio_in(ytarget[:, i*hop:(i+1)*hop])
        return self.get_generated_audio()
    
    def aggregate_top_activations(self, diag_fac=10, diag_len=10):
        """
        Aggregate activations from the top weight 0.1*self.P particles together
        to have them vote on the best activations

        Parameters
        ----------
        diag_fac: float
            Factor by which to promote probabilities of activations following
            activations chosen in the last steps
        diag_len: int
            Number of steps to look back for diagonal promotion
        """
        from scipy import sparse
        ## Step 1: Aggregate max particles
        PTop = int(self.neff_thresh)
        N = self.WCorpus.shape[1]
        ws = self.ws.cpu().numpy()
        idxs = np.argpartition(-ws, PTop)[0:PTop]
        states = self.states[idxs, :].cpu().numpy()
        ws = ws[idxs, None]*np.ones((1, states.shape[1]))
        states = states.flatten()
        ws = ws.flatten()
        probs = sparse.coo_matrix((ws, (states, np.zeros(states.size))), 
                                  shape=(N, 1)).toarray().flatten()
        
        ## Step 2: Promote states that follow the last state that was chosen
        promoted_idxs = np.zeros(N, dtype=int)
        for dc in range(1, min(diag_len, len(self.chosen_idxs))+1):
            last_state = self.chosen_idxs[-dc]+dc
            last_state = last_state[last_state < N]
            promoted_idxs[last_state] = 1
        probs[promoted_idxs > 0] *= diag_fac

        ## Step 3: Zero out last ones to prevent repeated activations
        for dc in range(1, min(self.r, len(self.chosen_idxs))+1):
            probs[self.chosen_idxs[-dc]] = 0
        return np.argpartition(-probs, self.pfinal)[0:self.pfinal]
    
    def process_window(self, x):
        """
        Run the particle filter for one step given the audio
        in one full window

        x: ndarray(n_channels, win)
            Window to process
        """
        if len(self.H)%10 == 0:
            print(".", end="", flush=True)
        ## Step 1: Do STFT of this window and sample from proposal distribution
        Vs = [self.feature_computer(self.win_samples*x[i, :]) for i in range(x.shape[0])]
        Vt = torch.concatenate(tuple(Vs))
        Vt = Vt.view(Vt.numel(), 1)
        ## Step 1b: Sample proposal indices based on the observation
        proposal_idxs = []
        VtNorm = 0
        if self.proposal_k > 0:
            Vtnp = Vt.cpu().numpy()
            VtNorm = np.sqrt(np.sum(Vtnp**2))
            if VtNorm > 0:
                proposal_idxs = self.proposal_tree.query((Vtnp/VtNorm).T, self.proposal_k, return_distance=False).flatten()
                proposal_idxs = self.loud_enough_idx_map[proposal_idxs]
                proposal_idxs = np.array(proposal_idxs, dtype=np.int32)
                proposal_idxs = torch.from_numpy(proposal_idxs).to(self.device)
        if self.proposal_k == 0 or VtNorm == 0:
            self.propagator.propagate(self.states)
        else:
            # Correction factor for proposal distribution
            self.ws *= self.propagator.propagate_proposal(self.states, proposal_idxs)

        ## Step 2: Apply the observation probability updates
        self.ws *= self.observer.observe(self.states, Vt)

        ## Step 3: Figure out the activations for this timestep
        ## by aggregating multiple particles near the top
        self.wsmax.append(torch.max(self.ws).item())
        if self.use_top_particle:
            top_idxs = self.states[torch.argmax(self.ws), :]
        else:
            top_idxs = self.aggregate_top_activations()
        self.chosen_idxs.append(top_idxs)

        h = do_KL_torch(self.WCorpus[:, top_idxs], self.WAlpha[top_idxs], Vt[:, 0], self.L)
        self.H.append(h)
        
        ## Step 4: Resample particles if effective number is too low
        self.ws /= torch.sum(self.ws)
        self.all_ws.append(self.ws.cpu().numpy())
        self.neff.append((1/torch.sum(self.ws**2)).item())
        if self.neff[-1] < self.neff_thresh:
            ## TODO: torch-ify stochastic universal sample
            self.num_resample += 1
            choices, _ = stochastic_universal_sample(self.all_ws[-1], len(self.ws))
            choices = torch.from_numpy(np.array(choices, dtype=int)).to(self.device)
            self.states = self.states[choices, :]
            self.ws = torch.ones(self.ws.shape).to(self.ws)/self.ws.numel()
        
        ## Step 5: Create and output audio samples for this window
        y = np.zeros_like(x)
        for i in range(x.shape[0]):
            y[i, :] = self.WSound[i][:, top_idxs].dot(h.cpu().numpy())
        self.audio_out(y)

        ## Step 6: Accumulate KL term for fit
        WH = torch.matmul(self.WCorpus[:, top_idxs], h)
        Vt = Vt.flatten()
        kl = (torch.sum(Vt*torch.log(Vt/WH) - Vt + WH)).item()
        self.fit += kl

    def plot_statistics(self):
        """
        Plot statistics about the activations that were chosen
        """
        p = self.states.shape[1]
        H = self.get_H()

        active_diffs = get_activations_diff(H, p)
        repeated_intervals = get_repeated_activation_itervals(H, p)
        
        plt.subplot(231)
        active_diffs = np.cumsum(active_diffs)
        avgwin = 10
        active_diffs = (active_diffs[avgwin::] - active_diffs[0:-avgwin])/avgwin
        t = np.arange(active_diffs.size)*self.win/(self.sr*2)
        plt.plot(t, active_diffs)
        plt.legend(["Particle Filter: Mean {:.3f}".format(np.mean(active_diffs))])
        plt.title("Activation Changes over Time, p={}, proposal_k={}".format(self.p, self.proposal_k))
        plt.xlabel("Time (Seconds)")

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
        plt.title("Neff (P={}, Median {:.2f}, Reampled {}x)".format(self.P, np.median(self.neff), self.num_resample))

        plt.subplot(236)
        diags = get_diag_lengths(H, p)
        plt.hist(diags, bins=np.arange(30))
        plt.legend(["Particle Filter Mean: {:.3f} ($p_d$={})".format(np.mean(diags), self.pd)])
        plt.xlabel("Diagonal Length")
        plt.ylabel("Counts")
        plt.title("Diagonal Lengths (Temperature {})".format(self.temperature))