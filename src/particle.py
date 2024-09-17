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
import matplotlib.pyplot as plt
from probutils import stochastic_universal_sample, do_KL, do_KL_torch, get_activations_diff_sparse, get_diag_lengths_sparse
from observer import Observer
from propagator import Propagator
from audioutils import get_windowed, hann_window
from audiofeatures import AudioFeatureComputer
import struct
import time
from threading import Lock
from tkinter import Tk, ttk, StringVar, OptionMenu

CORPUS_DB_CUTOFF = -50

class ParticleFilterChannel:
    """
    A particle filter for one channel
    """
    def reset_particles(self):
        """
        Randomly reset particles, each with the same weight 1/P
        """
        self.ws = np.array(np.ones(self.P)/self.P, dtype=np.float32) # Particle weights
        if self.device == "np":
            self.states = np.random.randint(self.N, size=(self.P, self.p))
        else:
            import torch
            self.ws = torch.from_numpy(self.ws).to(self.device) 
            self.states = torch.randint(self.N, size=(self.P, self.p), dtype=torch.int32).to(self.device) # Particles

    def reset_state(self):
        self.neff = [] # Number of effective particles over time
        self.wsmax = [] # Max weights over time
        self.ws = [] # Weights over time
        self.topcounts = [] 
        self.chosen_idxs = [] # Keep track of chosen indices
        self.H = [] # Activations of chosen indices
        self.reset_particles()
        self.all_ws = []
        self.fit = 0 # KL fit
        self.num_resample = 0

    def __init__(self, ycorpus, feature_params, particle_params, device, name="channel"):
        """
        ycorpus: ndarray(n_samples)
            Audio samples for the corpus for this channel
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
            r: int
                Repeated activations cutoff
            neff_thresh: float
                Number of effective particles below which to resample
            alpha: float
                L2 penalty for weights
            use_top_particle: bool
                If True, only take activations from the top particle at each step.
                If False, aggregate 
            target_shift: float
                Number of halfsteps by which to pitch shift the target
        }
        device: string
            Device string for torch
        name: string
            Name for this channel
        """
        tic = time.time()
        self.name = name
        win = feature_params["win"]
        sr = feature_params["sr"]
        self.p = particle_params["p"]
        self.P = particle_params["P"]
        self.pfinal = particle_params["pfinal"]
        self.pd = particle_params["pd"]
        self.temperature = particle_params["temperature"]
        self.L = particle_params["L"]
        self.r = particle_params["r"]
        self.neff_thresh = particle_params["neff_thresh"]
        self.alpha = particle_params["alpha"]
        self.use_top_particle = particle_params["use_top_particle"]
        self.device = device
        self.win = win
        self.sr = sr
        hop = win//2
        self.hop = hop
        self.wet = 1
        self.wet_mutex = Lock()
        # Store other channels whose parameters are coupled to this channel
        self.coupled_channels = [] 

        ## Step 1: Compute features for corpus
        self.target_shift = 0
        if "target_shift" in particle_params:
            self.target_shift = particle_params["target_shift"]
        self.target_shift_mutex = Lock()
        feature_params["device"] = device
        self.feature_computer = AudioFeatureComputer(**feature_params)
        print("Getting corpus windows for {}...".format(name), flush=True)
        self.WSound, WPowers = get_windowed(ycorpus, win, hann_window)
        # Corpus is analyzed with hann window
        self.win_samples = np.array(hann_window(win), dtype=np.float32)
        print("Computing corpus features for {}...".format(name), flush=True)
        WCorpus = self.feature_computer.get_spectral_features(self.WSound)
        self.WCorpus = WCorpus
        # Shrink elements that are too small
        self.WAlpha = self.alpha*np.array(WPowers <= CORPUS_DB_CUTOFF, dtype=np.float32)
        if self.device != "np":
            import torch
            self.WAlpha = torch.from_numpy(self.WAlpha).to(self.device)
        self.loud_enough_idx_map = np.arange(WCorpus.shape[1])[WPowers > CORPUS_DB_CUTOFF]
        print("{:.3f}% of corpus in {} is above loudness threshold".format(100*self.loud_enough_idx_map.size/WCorpus.shape[1], name))
        
        ## Step 2: Setup observer and propagator
        N = WCorpus.shape[1]
        self.N = N
        self.observer = Observer(self.p, WCorpus, self.WAlpha, self.L, self.temperature, device)
        self.propagator = Propagator(N, self.pd, device)
        self.reset_state()

        print("Finished setting up particle filter for {}: Elapsed Time {:.3f} seconds".format(name, time.time()-tic))
            
    def update_wet(self, value):
        with self.wet_mutex:
            self.wet = float(value)
            for c in self.coupled_channels:
                with c.wet_mutex:
                    c.wet = float(value)
    
    def update_target_shift(self, value):
        with self.target_shift_mutex:
            self.target_shift = float(value)
            self.target_shift_label.config(text="shift ({:.1f})".format(self.target_shift))
            for c in self.coupled_channels:
                with c.target_shift_mutex:
                    c.target_shift = float(value)

    def update_temperature(self, value):
        self.temperature = float(value)
        self.temp_label.config(text="temperature ({:.1f})".format(self.temperature))
        self.observer.update_temperature(self.temperature)
        for c in self.coupled_channels:
            c.temperature = float(value)
            c.observer.update_temperature(float(value))
    
    def update_pd(self, value):
        self.pd = float(value)
        self.pd_label.config(text="pd ({:.5f})".format(self.pd))
        self.propagator.update_pd(self.pd)
        for c in self.coupled_channels:
            c.pd = float(value)
            c.propagator.update_pd(float(value))

    def setup_gui(self, f, length):
        """
        Setup the GUI elements for the grid frame for this channel

        f: ttk.Frame
            Frame in which to put these widgets
        length: int
            Pixel width of each slider
        """
        row = 1

        ## Wet/dry slider
        self.wet_label = ttk.Label(f, text="Wet")
        self.wet_label.grid(column=1, row=row)
        self.wet_slider = ttk.Scale(f, from_=0, to=1, length=length, value=1, orient="horizontal", command=self.update_wet)
        self.wet_slider.grid(column=0, row=row)
        self.update_wet(self.wet)
        row += 1

        ## Pitch Shift slider
        self.target_shift_label = ttk.Label(f, text="Pitch Shift")
        self.target_shift_label.grid(column=1, row=row)
        self.target_shift_slider = ttk.Scale(f, from_=-12, to=12, length=length, value=0, orient="horizontal", command=self.update_target_shift)
        self.target_shift_slider.grid(column=0, row=row)
        self.update_target_shift(self.target_shift)
        row += 1

        ## Temperature slider
        self.temp_label = ttk.Label(f, text="temperature")
        self.temp_label.grid(column=1, row=row)
        self.temp_slider = ttk.Scale(f, from_=0, to=max(50, self.temperature*1.5), length=length, value=self.temperature, orient="horizontal", command=self.update_temperature)
        self.temp_slider.grid(column=0, row=row)
        self.update_temperature(self.temperature)
        row += 1

        ## pd slider
        self.pd_label = ttk.Label(f, text="pd")
        self.pd_label.grid(column=1, row=row)
        self.pd_slider = ttk.Scale(f, from_=0.5, to=1, length=length, value=self.pd, orient="horizontal", command=self.update_pd)
        self.pd_slider.grid(column=0, row=row)
        self.update_pd(self.pd)
        row += 1

        self.reset_button = ttk.Button(f, text="Reset Particles", command=self.reset_particles)
        self.reset_button.grid(column=0, row=row)
        row += 1

    def get_H(self, sparse=False):
        """
        Convert chosen_idxs and H into a numpy array with 
        activations in the proper indices

        Parameters
        ----------
        sparse: bool
            If True, return the sparse matrix directly

        Returns
        -------
        H: ndarray(N, T)
            Activations of the corpus over time
        """
        from scipy import sparse
        N = self.WCorpus.shape[1]
        T = len(self.H)
        vals = np.array(self.H).flatten()
        print("Min h: {:.3f}, Max h: {:.3f}".format(np.min(vals), np.max(vals)))
        rows = np.array(self.chosen_idxs, dtype=int).flatten()
        cols = np.array(np.ones((1, self.pfinal))*np.arange(T)[:, None], dtype=int).flatten()
        H = sparse.coo_matrix((vals, (rows, cols)), shape=(N, T))
        if not sparse:
            H = H.toarray()
        return H
    
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
        ## Step 1: Aggregate max particles
        PTop = int(self.neff_thresh)
        N = self.WCorpus.shape[1]
        ws = self.ws
        if self.device != "np":
            ws = ws.cpu().numpy()
        idxs = np.argpartition(-ws, PTop)[0:PTop]
        states = self.states[idxs, :]
        if self.device != "np":
            states = states.cpu().numpy()
        ws = ws[idxs]
        probs = {}
        for w, state in zip(ws, states):
            for idx in state:
                if not idx in probs:
                    probs[idx] = w
                else:
                    probs[idx] += w
        
        ## Step 2: Promote states that follow the last state that was chosen
        promoted_idxs = set([])
        for dc in range(1, min(diag_len, len(self.chosen_idxs))+1):
            last_state = self.chosen_idxs[-dc]+dc
            last_state = last_state[last_state < N]
            for idx in last_state:
                if not idx in promoted_idxs:
                    if idx in probs:
                        probs[idx] *= diag_fac
                    promoted_idxs.add(idx)

        ## Step 3: Zero out activations that happened over the last
        # r steps prevent repeated activations
        for dc in range(1, min(self.r, len(self.chosen_idxs))+1):
            for idx in self.chosen_idxs[-dc]:
                if idx in probs:
                    probs.pop(idx)
        
        ## Step 4: Choose top corpus activations
        idxs = np.array(list(probs.keys()), dtype=int)
        res = idxs
        if res.size <= self.pfinal:
            if res.size == 0:
                # If for some strange reason all the weights were 0
                res = np.random.randint(N, size=(self.pfinal,))
            else:
                while res.size < self.pfinal:
                    res = np.concatenate((res, res))[0:self.pfinal]
        else:
            # Common case: Choose top pfinal corpus positions by weight
            vals = np.array(list(probs.values()))
            res = idxs[np.argpartition(-vals, self.pfinal)[0:self.pfinal]]
        return res
    
    def get_Vt(self, x):
        """
        Compute windowed magnitude STFT of a chunk of audio

        Parameters
        ----------
        x: ndarray(win)
            Window to process
        
        Returns
        -------
        ndarray or torch (n_fft, 1)
            Spectrogram
        """
        with self.target_shift_mutex:
            Vt = self.feature_computer.get_spectral_features(self.win_samples*x, self.target_shift)
        if self.device == "np":
            Vt = np.reshape(Vt, (Vt.size, 1))
        else:
            Vt = Vt.view(Vt.numel(), 1)
        return Vt

    def do_particle_step(self, Vt):
        """
        Run the particle filter for one step given the audio
        in one full window for this channel, and figure out what
        the best activations are

        Vt: ndarray or torch (n_fft, 1)
            Spectrogram at this time

        Returns
        -------
        ndarray(p)
            Indices of best activations
        """
        ## Step 1: Propagate
        self.propagator.propagate(self.states)

        ## Step 2: Apply the observation probability updates
        self.ws *= self.observer.observe(self.states, Vt)

        ## Step 3: Figure out the activations for this timestep
        ## by aggregating multiple particles near the top
        if self.device == "np":
            self.wsmax.append(np.max(self.ws))
        else:
            import torch
            self.wsmax.append(torch.max(self.ws).item())
        if self.use_top_particle:
            top_idxs = self.states[torch.argmax(self.ws), :]
        else:
            top_idxs = self.aggregate_top_activations()
        self.chosen_idxs.append(top_idxs)
        
        ## Step 4: Resample particles if effective number is too low
        if self.device == "np":
            self.ws /= np.sum(self.ws)
            self.all_ws.append(np.array(self.ws))
            self.neff.append(1/np.sum(self.ws**2))
        else:
            import torch
            self.ws /= torch.sum(self.ws)
            self.all_ws.append(self.ws.cpu().numpy())
            self.neff.append((1/torch.sum(self.ws**2)).item())
        if self.neff[-1] < self.neff_thresh:
            ## TODO: torch-ify stochastic universal sample
            self.num_resample += 1
            choices, _ = stochastic_universal_sample(self.all_ws[-1], len(self.ws))
            choices = np.array(choices, dtype=int)
            if self.device != "np":
                import torch
                choices = torch.from_numpy(choices).to(self.device)
            self.states = self.states[choices, :]
            if self.device == "np":
                self.ws = np.ones(self.ws.shape)/self.ws.size
            else:
                import torch
                self.ws = torch.ones(self.ws.shape).to(self.ws)/self.ws.numel()

        return top_idxs
    
    def fit_activations(self, Vt, idxs):
        """
        Fit activations and mix audio

        Parameters
        ----------
        Vt: ndarray or torch (n_fft, 1)
            Spectrogram at this time
        idxs: ndarray(p, dtype=int)
            Indices of activations to use
        
        Returns
        -------
        y: ndarray(win)
            Resulting audio after running one step of the particle filter
        """
        ## Step 1: Compute activation weights
        kl_fn = do_KL
        if self.device != "np":
            kl_fn = do_KL_torch
        h = kl_fn(self.WCorpus[:, idxs], self.WAlpha[idxs], Vt[:, 0], self.L)
        hnp = h
        if self.device != "np":
            hnp = h.cpu().numpy()
        self.H.append(hnp)

        ## Step 2: Mix sound for this window
        y = self.WSound[:, idxs].dot(hnp)

        ## Step 3: Accumulate KL term for fit
        if self.device == "np":
            WH = self.WCorpus[:, idxs].dot(h)
        else:
            from torch import matmul
            WH = matmul(self.WCorpus[:, idxs], h)
        Vt = Vt.flatten()
        # Take care of numerical issues
        Vt = Vt[WH > 0]
        WH = WH[WH > 0]
        WH = WH[Vt > 0]
        Vt = Vt[Vt > 0]
        if self.device == "np":
            kl = np.sum(Vt*np.log(Vt/WH) - Vt + WH)
        else:
            import torch
            kl = (torch.sum(Vt*torch.log(Vt/WH) - Vt + WH)).item()
        self.fit += kl

        return y

class ParticleAudioProcessor:
    """
    A class that has the following responsibilities:
        * Handles input/output, possibly using the microphone
        * Coordinates particle filters for each channel
        * Creates an over-arching GUI when doing real time input
        * Provides a method to plot statistics of the particle filters
          for each channel once a run has finished
    """
    def reset_state(self):
        """
        Reset all of the audio buffers and particle filters
        """
        # Keep track of time to process each frame
        self.frame_times = [] 
        # Setup a circular buffer that receives in hop samples at a time
        self.buf_in  = np.zeros((self.n_channels, self.win), dtype=np.float32)
        # Setup an output buffer that doubles in size like an arraylist
        self.buf_out = np.zeros((self.n_channels, self.sr*60*10), dtype=np.float32)
        # Variable to track beginning of the window that's just been written to the output buffer
        self.output_idx = 0 
        for c in self.channels:
            c.reset_state()

    def __init__(self, ycorpus, feature_params, particle_params, device, use_mic=False, couple_channels=True):
        """
        ycorpus: ndarray(n_channels, n_samples)
            Audio samples for the corpus, possibly multi-channel
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
            r: int
                Repeated activations cutoff
            neff_thresh: float
                Number of effective particles below which to resample
            alpha: float
                L2 penalty for weights
            use_top_particle: bool
                If True, only take activations from the top particle at each step.
                If False, aggregate 
            target_shift: float
                Number of halfsteps by which to pitch shift the target
        }
        device: string
            Device string for torch
        use_mic: bool
            If true, use the microphone
        couple_channels: bool
            If true, only run a particle filter on the first channel, and use the
            same corpus elements on the other channels. (Default true)
            Otherwise, run individual particle filters on each channel
        """
        self.device = device
        self.use_mic = use_mic
        self.couple_channels = couple_channels
        self.sr = feature_params["sr"]
        self.win = feature_params["win"]
        self.hop = self.win//2
        feature_params["device"] = device
        self.n_channels = ycorpus.shape[0]
        self.channels = [ParticleFilterChannel(ycorpus[i, :], feature_params, particle_params, device, name="channel {}".format(i)) for i in range(ycorpus.shape[0])]
        if self.couple_channels:
            for c in self.channels[1:]:
                self.channels[0].coupled_channels.append(c)
        self.reset_state()
        if use_mic:
            from pyaudio import PyAudio
            self.recorded_audio = []
            self.frame_mutex = Lock()
            self.processing_frame = False
            self.audio = PyAudio()
            self.recording_started = False
            self.recording_finished = False
            self.setup_gui()

    def setup_gui(self):
        """
        Setup the GUI that's active for microphone recording, including
        widgets for each channel's parameters
        """
        ## Step 1: Setup menus for recording and device selection
        self.tk_root = Tk()
        f = ttk.Frame(self.tk_root, padding=10)
        f.grid()
        row = 0
        ttk.Label(f, text="The Concatenator Real Time!").grid(column=0, row=row)
        row += 1
        self.record_button = ttk.Button(f, text="Start Recording", command=self.start_audio_recording)
        self.record_button.grid(column=0, row=row)
        self.device2obj = {}
        devices = []
        for i in range(self.audio.get_device_count()):
            # https://forums.raspberrypi.com/viewtopic.php?t=71062
            dev = self.audio.get_device_info_by_index(i)
            if dev["maxInputChannels"] > 0:
                dev["i"] = i
                devices.append(dev['name'])
                self.device2obj[dev['name']] = dev
        self.dev_clicked = StringVar()
        self.dev_clicked.set(devices[0])
        self.dev_drop = OptionMenu(f, self.dev_clicked, *devices)
        self.dev_drop.grid(column=1, row=row)
        row += 1

        ## Step 2: Setup widgets for each channel
        width = 800
        for i, c in enumerate(self.channels):
            if i == 0 or not self.couple_channels:
                framei = ttk.Frame(f)
                framei.grid(row=row, column=i, padx=10, pady=5)
                c.setup_gui(framei, width//len(self.channels))

        ## Step 3: Start Tkinter loop!
        self.tk_root.mainloop()

    def start_audio_recording(self):
        """
        Setup the audio stream for microphone recording
        """
        from pyaudio import paFloat32
        dev = self.device2obj[self.dev_clicked.get()]
        self.mic_channels = min(dev["maxInputChannels"], self.n_channels)
        ## Step 1: Run one frame with dummy data to precompile all kernels
        ## Use high amplitude random noise to get it to jump around a lot
        hop = self.win//2
        bstr = np.array(np.random.rand(hop*self.mic_channels), dtype=np.float32)
        bstr = struct.pack("<"+"f"*hop*self.mic_channels, *bstr)
        for _ in range(20):
            self.audio_in(bstr)
        self.reset_state()
        self.recorded_audio = []
        self.output_idx = 0
        self.buf_in *= 0
        self.buf_out *= 0

        ## Step 2: Setup stream
        self.stream = self.audio.open(format=paFloat32, 
                            frames_per_buffer=self.hop, 
                            channels=self.mic_channels, 
                            rate=self.sr, 
                            input_device_index = dev["i"],
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
    
    def get_generated_audio(self):
        """
        Return the audio that's been generated

        Returns
        -------
        ndarray(n_samples, 2)
            Generated audio
        """
        ret = self.buf_out[:, 0:self.output_idx+self.win]
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

    def accumulate_next_window(self, x):
        """
        Incorporate a new window into the output audio buffer
        After this method is run, we are ready to output hop more samples

        x: ndarray(n_channels, win)
            Windowed multi-channel audio
        """
        win = self.win
        N = self.buf_out.shape[1]
        if N < self.output_idx + self.win:
            # Double size
            new_out = np.zeros((self.buf_out.shape[0], N*2), dtype=np.float32)
            new_out[:, 0:N] = self.buf_out
            self.buf_out = new_out
        for i in range(self.n_channels):
            with self.channels[i].wet_mutex:
                wet = self.channels[i].wet
                self.buf_out[i, self.output_idx:self.output_idx+win] += x[i, :]*wet + self.buf_in[i, :]*(1-wet)

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
        status_ret = None
        if self.use_mic:
            from pyaudio import paContinue
            status_ret = paContinue
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
                return s, paContinue
            fmt = "<"+"f"*(self.mic_channels*self.win//2)
            x = np.array(struct.unpack(fmt, s), dtype=np.float32)
            x = np.reshape(x, (x.size//self.mic_channels, self.mic_channels)).T
            self.recorded_audio.append(x)
        else:
            x = s
        hop = self.win//2
        self.buf_in[:, 0:hop] = self.buf_in[:, hop:]
        self.buf_in[:, hop:] = x
        y = np.zeros((self.n_channels, self.win))
        idxs = []
        for i, c in enumerate(self.channels):
            # Run each particle filter on its channel of audio
            Vt = c.get_Vt(self.buf_in[i, :])
            if i == 0 or not self.couple_channels:
                idxs = c.do_particle_step(Vt)
            y[i, :] = c.fit_activations(Vt, idxs)
        self.accumulate_next_window(y)
        # Record elapsed time
        elapsed = time.time()-tic
        self.frame_times.append(elapsed)
        # Output the audio that's ready
        ret = self.buf_out[:, self.output_idx:self.output_idx+hop].T
        ret = ret.flatten()
        # Move forward by one hop
        self.output_idx += hop
        if (self.output_idx//hop)%10 == 0:
            print(".", end="", flush=True)
        if self.use_mic:
            with self.frame_mutex:
                self.processing_frame = False
        return struct.pack("<"+"f"*ret.size, *ret), status_ret
         
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

    def plot_statistics(self):
        """
        Plot statistics about the activations that were chosen
        """
        p = self.channels[0].states.shape[1]
        channels_to_plot = self.channels
        if self.couple_channels:
            channels_to_plot = channels_to_plot[0:1]
        Hs = [c.get_H(sparse=True) for c in channels_to_plot]
        all_active_diffs = [get_activations_diff_sparse(H.row, H.col, p) for H in Hs]
        
        plt.subplot2grid((2, 3), (0, 0), colspan=2)
        legend = []
        for (active_diffs, c) in zip(all_active_diffs, channels_to_plot):
            t = np.arange(active_diffs.size)*self.win/(self.sr*2)
            plt.plot(t, active_diffs, linewidth=0.5)
            legend.append("{}: Mean {:.3f}".format(c.name, np.mean(active_diffs)))
        plt.legend(legend)
        plt.title("Activation Changes over Time, p={}".format(p))
        plt.xlabel("Time (Seconds)")

        plt.subplot(233)
        legend = []
        for (active_diffs, c) in zip(all_active_diffs, channels_to_plot):
            plt.hist(active_diffs, bins=np.arange(p+2), alpha=0.5)
            legend.append(c.name)
        plt.legend(legend)
        plt.title("Activation Changes Histogram")
        plt.xlabel("Number of Activations Changed")
        plt.ylabel("Counts")

        plt.subplot(234)
        legend = []
        for c in channels_to_plot:
            plt.plot(c.wsmax)
            legend.append("{}: {:.2f}".format(c.name, c.fit))
        plt.legend(legend)
        plt.title("Max Probability And Overall Fit")
        plt.xlabel("Timestep")

        plt.subplot(235)
        legend = []
        for c in channels_to_plot:
            plt.plot(c.neff)
            legend.append("{} Med {:.2f}, Resampled {}x".format(c.name, np.median(c.neff), c.num_resample))
        plt.legend(legend)
        plt.xlabel("Timestep")
        plt.title("Neff (P={})".format(channels_to_plot[0].P))

        plt.subplot(236)
        legend = []
        for c, H in zip(channels_to_plot, Hs):
            diags = get_diag_lengths_sparse(H.row, H.col)
            legend.append("{} Mean: {:.3f}".format(c.name, np.mean(diags)))
            plt.hist(diags, bins=np.arange(30), alpha=0.5)
        plt.legend(legend)
        plt.xlabel("Diagonal Length ($p_d$={})".format(channels_to_plot[0].pd))
        plt.ylabel("Counts")
        plt.title("Diagonal Lengths (Temperature {})".format(channels_to_plot[0].temperature))