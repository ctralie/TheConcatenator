import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import pickle
import time
from scipy import sparse
sys.path.append("../src")
from driedger import *
from probutils import *
from audioutils import *


## Step 1a: Initialize fixed Driedger parameters
c = 3 # Diagonal
r = 3 # Repeated activation
KL_BASIC_p = 10
## Step 1b: intialize acoustic parameters
sr = 44100
win = 2048
hop = win//2
min_freq = 0
max_freq = 8000
kmin = max(0, int(win*min_freq/sr)+1)
kmax = min(int(win*max_freq/sr)+1, win//2)


## Step 2: Initialize corpus
#corpusname = "Bees"
#ycorpus = load_corpus("../corpus/Bees_Buzzing.mp3", sr, True)
#stereo = True

corpusname = "EdenVIP2"
ycorpus = load_corpus("../corpus/EdenVIP2", sr, True)
stereo = True

WSoundL, _ = get_windowed(ycorpus[0, :], win)
WSoundR, _ = get_windowed(ycorpus[1, :], win)
WL = np.abs(np.fft.fft(WSoundL, axis=0)[kmin:kmax, :])
WR = np.abs(np.fft.fft(WSoundR, axis=0)[kmin:kmax, :])
W = np.concatenate((WL, WR), axis=0)


### Step 3: Setup targets for this batch
N = 1000
n_batches = int(sys.argv[1])
batch_num = int(sys.argv[2])
files = glob.glob("../target/fma_small/*/*.mp3")
np.random.seed(len(files))
files = sorted(files)
files = [files[idx] for idx in np.random.permutation(len(files))[0:N]]
files = sorted(files)
outfilenames = [f[0:-4] + "_driedger_" + corpusname + ".pkl" for f in files]
not_finished = [not os.path.exists(f) for f in outfilenames]
files = [f for (f, n) in zip(files, not_finished) if n]
outfilenames = [f for (f, n) in zip(outfilenames, not_finished) if n]
N = len(files)
chunk = int(np.ceil(N/n_batches))
i = batch_num
files = files[i*chunk:(i+1)*chunk]
outfilenames = outfilenames[i*chunk:(i+1)*chunk]

print("Processing {} files in batch {} of {}".format(len(files), batch_num+1, n_batches))

for target, outfilename in zip(files, outfilenames):
    print("Doing", outfilename)
    res = {}
    ytarget = load_corpus(target, sr=sr, stereo=stereo)
    W1L, _ = get_windowed(ytarget[0, :], win)
    W1R, _ = get_windowed(ytarget[1, :], win)
    VL = np.abs(np.fft.fft(W1L, axis=0)[kmin:kmax, :])
    VR = np.abs(np.fft.fft(W1R, axis=0)[kmin:kmax, :])
    V = np.concatenate((VL, VR), axis=0)


    for p in [0, 5, 10]:
        tic = time.time()
        if p == 0:
            H = get_basic_KL_activations(V, W, L=50, verbose=True)
        else:
            H = get_musaic_activations(V, W, L=50, r=r, p=p, c=c, verbose=True)
        elapsed = time.time()-tic
        print("Elapsed time: {:.3f}".format(elapsed))
        WH = W.dot(H)
        Vi = V[WH > 0]
        WH = WH[WH > 0]
        WH = WH[Vi > 0]
        Vi = Vi[Vi > 0]
        fit = np.sum(Vi*np.log(Vi/WH) - Vi + WH)
        if p == 0:
            # Sparsify basic KL
            HSort = -np.partition(-H, KL_BASIC_p, axis=0)[KL_BASIC_p, :]
            H = H*(H > HSort[None, :])
        H = sparse.coo_matrix(H)
        res["p{}".format(p)] = dict(
            fit=fit,
            row=H.row,
            col=H.col,
            data=H.data,
            elapsed=elapsed
        )

        pickle.dump(res, open(outfilename, "wb"))