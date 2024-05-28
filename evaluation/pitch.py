from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import sparse
import sys
sys.path.append("../src")
from particle import *
from bayes import *
from probutils import *
from audioutils import *
import time
import pickle
import os
import glob
import torch
import torchcrepe

# For crepe
fmin = 50
fmax = 1000
batch_size = 4096
hop_crepe = 1024

sr = 44100
win = 2048
hop = win//2
p = 3
r = 3 # Repeated activation
stereo = False

pparticle = p
pd = 0.95
temperature = 10
L = 10
P = 10000
gamma = 0
neff_thresh = 0.1*P
min_freq = 0
max_freq = 8000
tic = time.time()


feature_params = dict(
    win=win,
    sr=sr,
    min_freq=min_freq,
    max_freq=max_freq,
    use_stft=True,
    use_mel=False,
    mel_bands=40
)
particle_params = dict(
    p=p,
    pfinal=p,
    pd=pd,
    temperature=temperature,
    L=L,
    P=P,
    gamma=gamma,
    r=r,
    neff_thresh=neff_thresh,
    proposal_k=0,
    use_top_particle=False,
    alpha=1
)

use_mic = False
ycorpus = load_corpus("../corpus/UIowa/Woodwinds", sr, stereo=stereo)

for mel in [False, True]:
    if mel:
        particle_params["mel"] = True
        particle_params["stft"] = False
    else:
        particle_params["mel"] = False
        particle_params["stft"] = True
    for P in [100, 1000, 10000]:
        particle_params["P"] = P
        particle_params["neff_thresh"] = 0.1*P
        pf = ParticleAudioProcessor(ycorpus, feature_params, particle_params, 'cuda')
        
        for f in glob.glob("../target/MDB-stem-synth/audio_stems/*.wav"):
            pf.reset_state()
            ## Step 1: Load file
            prefix = f[0:-4]
            if mel:
                prefix = prefix + "_mel"
            outname = "{}_P{}.pkl".format(prefix, P)
            if os.path.exists(outname):
                print("Skipping", outname)
                continue

            ytarget = load_corpus(f, sr=sr, stereo=stereo)

            ## Step 2: Compute CREPE on raw audio
            if P == 100 and mel == False:
                pitch = torchcrepe.predict(torch.from_numpy(ytarget.flatten()).view((1, ytarget.size)),sr,hop_crepe, fmin, fmax,'full',batch_size=batch_size,device="cuda").cpu().numpy().flatten()
                pickle.dump(pitch, open("{}_raw.pkl".format(prefix), "wb"))

            ## Step 3: Run concatenator and compute pitch on result
            tic = time.time()
            ygen = pf.process_audio_offline(ytarget)
            print("Elapsed time particle filter:", time.time()-tic)
            H = sparse.coo_matrix(pf.get_H())
            tic = time.time()
            pitch = torchcrepe.predict(torch.from_numpy(ygen.flatten()).view((1, ygen.size)),sr,hop_crepe, fmin, fmax,'full',batch_size=batch_size,device="cuda").cpu().numpy().flatten()
            res = dict(
                pitch=pitch,
                fit=pf.fit,
                row=H.row,
                col=H.col,
                data=H.data,
            )
            pickle.dump(res, open(outname, "wb"))
            print("Elapsed time pitch", time.time()-tic)
