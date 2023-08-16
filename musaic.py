"""
Programmer: Chris Tralie
Purpose: To serve as an entry point for my new musaicing technique
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from audioutils import *
from letitbee import *
from particle import *
from bayes import *
from probutils import *
import librosa
import time
from scipy.io import wavfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help="Path to audio file or directory for source sounds")
    parser.add_argument('--target', type=str, required=True, help="Path to audio file for target sound")
    parser.add_argument('--result', type=str, required=True, help="Path to wav file to which to save the result")
    parser.add_argument('--sr', type=int, default=44100, help="Sample rate")
    parser.add_argument('--winSize', type=int, default=2048, help="Window Size in samples")
    parser.add_argument('--hopSize', type=int, default=1024, help="Hop Size in samples")
    parser.add_argument('--stereo', type=int, default=1, help="If 1, use stereo.  If 0, use mono")
    parser.add_argument('--use_gpu', type=int, default=1, help="If 1 (default), use GPU.  If 0, use CPU")
    parser.add_argument('--use_mel', type=int, default=0, help="If 1 use a mel-spaced spectrogram instead of a regular spectrogram")
    #parser.add_argument('--shiftrange', type=int, default=0, help="The number of halfsteps below and above which to shift the sound")
    parser.add_argument('--r', type=int, default=7, help="Width of the repeated activation filter")
    parser.add_argument('--p', type=int, default=10, help="Number of simultaneous activations")
    parser.add_argument('--pd', type=float, default=0.99, help="Probability of sticking to an activation (0 is no stick, closer to 1 is longer continuous activations)")
    parser.add_argument('--L', type=int, default=10, help="Number of KL iterations")
    parser.add_argument('--particles', type=int, default=2000, help="Number of particles in the particle filter")
    parser.add_argument('--temperature', type=float, default=100, help="Target importance.  Higher values mean activations will jump around more to match the target.")
    parser.add_argument('--saveplots', type=int, default=0, help='Save plots of iterations to disk')
    opt = parser.parse_args()

    ytarget, sr = librosa.load(opt.target, sr=opt.sr, mono=not opt.stereo)
    if opt.stereo and len(ytarget.shape) == 1:
        ytarget = np.array([ytarget, ytarget])

    print("Loading corpus...")
    ycorpus = load_corpus(opt.corpus, opt.sr, opt.stereo)

    if not opt.stereo:
        ytarget = np.array([ytarget, ytarget])
        ycorpus = np.array([ycorpus, ycorpus])
    
    hop = opt.hopSize
    win = opt.winSize
   
    W1L = get_windowed(ytarget[0, :], hop, win)
    W1R = get_windowed(ytarget[1, :], hop, win)
    VL = np.abs(np.fft.fft(W1L, axis=0)[0:win//2+1, :])
    VR = np.abs(np.fft.fft(W1R, axis=0)[0:win//2+1, :])
    if opt.use_mel:
        M = get_mel_filterbank(win, sr, -24, 50)
        V = np.concatenate((M.dot(VL), M.dot(VR)), axis=0)
    else:
        V = np.concatenate((VL[1:win//4, :], VR[1:win//4, :]), axis=0)

    WSoundL = get_windowed(ycorpus[0, :], hop, win)
    WSoundR = get_windowed(ycorpus[1, :], hop, win)
    WL = np.abs(np.fft.fft(WSoundL, axis=0)[0:win//2+1, :])
    WR = np.abs(np.fft.fft(WSoundR, axis=0)[0:win//2+1, :])
    if opt.use_mel:
        W = np.concatenate((M.dot(WL), M.dot(WR)), axis=0)
    else:
        W = np.concatenate((WL[1:win//4, :], WR[1:win//4, :]), axis=0)
    
    p = opt.p
    pd = opt.pd
    temperature = opt.temperature
    L = opt.L
    P = opt.particles
    r = opt.r
    neff_thresh = 0.1*P

    if p == 1:
        print("Finished setting up corpus; doing Bayes filter for p=1")
        tic = time.time()
        H, wsmax, neff = get_bayes_musaic_activations(V, W, p, pd, temperature, L, r)
        print("Elapsed time bayes filter: {:.3f}".format(time.time()-tic))
    else:
        print("Finished setting up corpus; doing particle filter")
        tic = time.time()
        H, wsmax, neff = get_particle_musaic_activations(V, W, p, p, pd, temperature, L, P, r=r, neff_thresh=neff_thresh, use_gpu=(opt.use_gpu==1))
        print("Elapsed time particle filter: {:.3f}".format(time.time()-tic))

    active_diffs = get_activations_diff(H, p)
    repeated_intervals = get_repeated_activation_itervals(H, p)

    WH = W.dot(H)
    fit = np.sum(V*np.log(V/WH) - V + WH)
    yL = do_windowed_sum(WSoundL, H, win, hop)
    yR = do_windowed_sum(WSoundR, H, win, hop)

    y = np.array([yL, yR]).T
    y = y/np.max(y)
    wavfile.write(opt.result, sr, y)



    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.plot(active_diffs)
    plt.legend(["Particle Filter: Mean {:.3f}".format(np.mean(active_diffs))])
    plt.title("Activation Changes over Time, Temperature {}".format(temperature))
    plt.xlabel("Timestep")
    plt.subplot(232)
    plt.hist(active_diffs, bins=np.arange(30))
    plt.title("Activation Changes Histogram")
    plt.xlabel("Number of Activations Changed")
    plt.ylabel("Counts")
    plt.legend(["Ground Truth", "Particle Filter"])
    plt.subplot(233)
    plt.hist(repeated_intervals, bins=np.arange(30))
    plt.title("Repeated Activations Histogram, r={}".format(r))
    plt.xlabel("Repeated Activation Distance")
    plt.ylabel("Counts")
    plt.legend(["Ground Truth", "Particle Filter"])
    plt.subplot(234)
    plt.plot(wsmax)
    plt.title("Max Probability, Overall Fit: {:.3f}".format(fit))
    plt.xlabel("Timestep")
    plt.subplot(235)
    plt.plot(neff)
    plt.xlabel("Timestep")
    plt.title("Number of Effective Particles (Median {:.2f})".format(np.median(neff)))
    plt.subplot(236)
    diags = get_diag_lengths(H, p)
    plt.hist(diags, bins=np.arange(30))
    plt.legend(["Particle Filter Mean: {:.3f} ($p_d$={})".format(np.mean(diags), pd)])
    plt.xlabel("Diagonal Length")
    plt.ylabel("Counts")
    plt.title("Diagonal Lengths")
    plt.tight_layout()
    plt.savefig("{}.svg".format(opt.result), bbox_inches='tight')