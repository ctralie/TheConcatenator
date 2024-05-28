import numpy as np
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


def do_batch_with_params(pfiles, feature_params, particle_params):
    id = "_p{}_temp{}_P{}_pd{}_proposalK{}_".format(
        particle_params["p"],
        particle_params["temperature"],
        particle_params["P"],
        particle_params["pd"],
        particle_params["proposal_k"]
    )
    if (particle_params["use_mel"]):
        id += "mel_"

    outfilenames = [f[0:-4] + id + corpusname + ".pkl" for f in pfiles]
    not_finished = [not os.path.exists(f) for f in outfilenames]
    files = [f for (f, n) in zip(pfiles, not_finished) if n]
    outfilenames = [f for (f, n) in zip(outfilenames, not_finished) if n]

    pf = ParticleAudioProcessor(ycorpus, feature_params, particle_params, 'cuda')

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
    mel_bands=40
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


### Step 2: Setup targets for this batch
N = 1000
files = glob.glob("../target/fma_small/*/*.mp3")
np.random.seed(len(files))
files = sorted(files)
files = [files[idx] for idx in np.random.permutation(len(files))[0:N]]
files = sorted(files)

## Step 3: Initialize corpus
for experiment in [
    dict(corpusname="Bees", stereo=True, path="../corpus/Bees_Buzzing.mp3"),
    dict(corpusname="EdenVIP2", stereo=True, path="../corpus/EdenVIP2"),
    dict(corpusname="Woodwinds", stereo=False, path="../corpus/UIowa/Woodwinds")
]:
    corpusname = experiment["corpusname"]
    stereo = experiment["stereo"]
    ycorpus = load_corpus(experiment["path"], sr, stereo)

    ## Slice 1: Different numbers of particles and proposal distribution
    ## Plots: Show Driedger p=0,5,10 against these
    print("Doing slice 1")
    particle_params["pd"] = 0.95
    particle_params["temperature"] = 10
    particle_params["p"] = 5
    particle_params["pfinal"] = 5
    particle_params["use_stft"] = False
    particle_params["use_mel"] = False
    for P in [100, 1000, 10000]:
        particle_params["P"] = P
        particle_params["neff_thresh"] = 0.1*P
        proposal_ks = [0]
        if P == 100:
            # Only use proposal distribution in one specific case
            proposal_ks.append(10) 
        for proposal_k in proposal_ks:
            particle_params["proposal_k"] = proposal_k
            do_batch_with_params(files, feature_params, particle_params)
    ## Slice 1b: Do one run with p=10 for 1000 particles
    print("Doing slice 1b")
    P = 1000
    particle_params["P"] = P
    particle_params["neff_thresh"] = 0.1*P
    particle_params["proposal_k"] = 0
    particle_params["p"] = 10
    particle_params["pfinal"] = 10
    do_batch_with_params(files, feature_params, particle_params)


    ## Slice 2: A range of pd
    print("Doing slice 2")
    particle_params["temperature"] = 10
    particle_params["p"] = 5
    particle_params["pfinal"] = 5
    P = 1000
    particle_params["P"] = P
    particle_params["neff_thresh"] = 0.1*P
    particle_params["proposal_k"] = 0
    for pd in [0.9, 0.95, 0.99, 0.5]:
        particle_params["pd"] = pd
        do_batch_with_params(files, feature_params, particle_params)


    ## Slice 3: A range of temperature
    print("Doing slice 3")
    particle_params["pd"] = 0.95
    particle_params["p"] = 5
    particle_params["pfinal"] = 5
    P = 1000
    particle_params["P"] = P
    particle_params["neff_thresh"] = 0.1*P
    particle_params["proposal_k"] = 0
    for temperature in [1, 10, 50]:
        particle_params["temperature"] = temperature
        do_batch_with_params(files, feature_params, particle_params)
