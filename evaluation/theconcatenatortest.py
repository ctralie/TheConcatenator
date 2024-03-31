import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import pickle
import time
from scipy import sparse
sys.path.append("../src")
from particle import *
from probutils import *
from audioutils import *


## Step 1: intialize parameters
sr = 44100
win = 2048
hop = win//2
stereo = True
P = 1000
feature_params = dict(
    win=win,
    sr=sr,
    min_freq=0,
    max_freq=8000,
    use_stft=True,
    use_mel=False,
    mel_bands=40,
    use_zcs=False,
)
particle_params = dict(
    p=5,
    pfinal=5,
    pd=0.9,
    temperature=1,
    L=10,
    P=P,
    r=3,
    neff_thresh=0.1*P,
    proposal_k=0,
    alpha=0.1,
    use_top_particle=False
)

## Step 2: Initialize corpus
corpusname = "EdenVIP2"
ycorpus = load_corpus("../corpus/EdenVIP2", sr, True)


### Step 3: Setup targets for this batch
N = 1000
files = glob.glob("../target/fma_small/*/*.mp3")
np.random.seed(len(files))
files = sorted(files)
files = [files[idx] for idx in np.random.permutation(len(files))[0:N]]
files = sorted(files)


def do_batch_with_params(pfiles, feature_params, particle_params):
    id = "_p{}_temp{}_P{}_pd{}_proposalK{}_".format(
        particle_params["p"],
        particle_params["temperature"],
        particle_params["P"],
        particle_params["pd"],
        particle_params["proposal_k"]
    )
    outfilenames = [f[0:-4] + id + corpusname + ".pkl" for f in pfiles]
    not_finished = [not os.path.exists(f) for f in outfilenames]
    files = [f for (f, n) in zip(pfiles, not_finished) if n]
    outfilenames = [f for (f, n) in zip(outfilenames, not_finished) if n]

    pf = ParticleFilter(ycorpus, feature_params, particle_params, 'cuda')

    for target, outfilename in zip(files, outfilenames):
        print(outfilename)
        pf.reset_state()
        ytarget = load_corpus(target, sr=sr, stereo=stereo)
        
        tic = time.time()
        pf.process_audio_offline(ytarget)
        elapsed = time.time()-tic
        print("Elapsed time: {:.3f}".format(elapsed))
        H = sparse.coo_matrix(pf.get_H())
        res = dict(
            fit=pf.fit,
            row=H.row,
            col=H.col,
            data=H.data,
            elapsed=elapsed
        )
        print("fit", pf.fit)

        pickle.dump(res, open(outfilename, "wb"))

for pd in [0.9, 0.95, 0.99, 0.5]:
    particle_params["pd"] = pd
    for p in [5, 10]:
        particle_params["p"] = p
        particle_params["pfinal"] = p
        for temperature in [1, 10, 50]:
            particle_params["temperature"] = temperature
            for P in [100, 1000, 10000]:
                particle_params["P"] = P
                particle_params["neff_thresh"] = 0.1*P
                for proposal_k in [0, 10]:
                    particle_params["proposal_k"] = proposal_k
                    if P != 10000 or (p == 5 and temperature == 10 and pd == 0.9 and proposal_k == 0):
                        # Since 10000 is so expensive, only look at it with p=5, temperature=10, pd=0.9, and no proposal k
                        do_batch_with_params(files, feature_params, particle_params)
