"""
Programmer: Chris Tralie
Purpose: To serve as an entry point for my new musaicing technique
"""
import argparse
import numpy as np
from audioutils import load_corpus
from particle import ParticleFilter
import librosa
import time
from scipy.io import wavfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help="Path to audio file or directory for source sounds")
    parser.add_argument('--target', type=str, required=True, help="Path to audio file for target sound, or \"mic\" if using the microphone to do real time")
    parser.add_argument('--result', type=str, required=True, help="Path to wav file to which to save the result")
    parser.add_argument('--sr', type=int, default=44100, help="Sample rate")
    parser.add_argument('--winSize', type=int, default=2048, help="Window Size in samples")
    parser.add_argument('--stereo', type=int, default=1, help="If 1, use stereo.  If 0, use mono")
    parser.add_argument('--use_gpu', type=int, default=1, help="If 1 (default), use GPU.  If 0, use CPU")
    #parser.add_argument('--shiftrange', type=int, default=0, help="The number of halfsteps below and above which to shift the sound")
    parser.add_argument('--r', type=int, default=7, help="Width of the repeated activation filter")
    parser.add_argument('--p', type=int, default=10, help="Number of simultaneous activations")
    parser.add_argument('--pd', type=float, default=0.99, help="Probability of sticking to an activation (0 is no stick, closer to 1 is longer continuous activations)")
    parser.add_argument('--L', type=int, default=10, help="Number of KL iterations")
    parser.add_argument('--particles', type=int, default=2000, help="Number of particles in the particle filter")
    parser.add_argument('--gamma', type=float, default=0, help="Cosine similarity cutoff for proposal distribution")
    parser.add_argument('--temperature', type=float, default=100, help="Target importance.  Higher values mean activations will jump around more to match the target.")
    parser.add_argument('--saveplots', type=int, default=1, help='Save plots of iterations to disk')
    opt = parser.parse_args()

    ytarget, sr = librosa.load(opt.target, sr=opt.sr, mono=not opt.stereo)
    if opt.stereo and len(ytarget.shape) == 1:
        ytarget = np.array([ytarget, ytarget])

    print("Loading corpus...")
    ycorpus = load_corpus(opt.corpus, opt.sr, opt.stereo)

    if not opt.stereo:
        ytarget = np.array([ytarget, ytarget])
        ycorpus = np.array([ycorpus, ycorpus])

    print("Finished setting up corpus; doing particle filter")
    tic = time.time()
    pf = ParticleFilter(ycorpus=ycorpus, win=opt.winSize, p=opt.p, pfinal=opt.p, pd=opt.pd, temperature=opt.temperature, L=opt.L, P=opt.particles, gamma=opt.gamma, r=opt.r, neff_thresh=0.1*opt.particles, use_gpu=(opt.use_gpu==1))
    hop = opt.winSize//2
    for i in range(0, ytarget.shape[1]//hop):
        pf.audio_in(ytarget[:, i*hop:(i+1)*hop])
    print("Elapsed time particle filter: {:.3f}".format(time.time()-tic))

    wavfile.write(opt.result, sr, pf.get_generated_audio())

    if opt.saveplots == 1:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        pf.plot_statistics()
        plt.tight_layout()
        plt.savefig("{}.svg".format(opt.result), bbox_inches='tight')