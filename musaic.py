"""
Programmer: Chris Tralie
Purpose: To serve as an entry point for my new musaicing technique
"""
import argparse
import sys
sys.path.append("src")
from audioutils import load_corpus
from particle import ParticleFilter
import time
from scipy.io import wavfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help="Path to audio file or directory for source sounds")
    parser.add_argument('--target', type=str, required=True, help="Path to audio file for target sound, or \"mic\" if using the microphone to do real time")
    parser.add_argument('--result', type=str, required=True, help="Path to wav file to which to save the result")
    parser.add_argument('--winSize', type=int, default=2048, help="Window Size in samples")
    parser.add_argument('--sr', type=int, default=44100, help="Sample rate")
    parser.add_argument('--minFreq', type=int, default=50, help="Minimum frequency to use (in hz)")
    parser.add_argument('--maxFreq', type=int, default=10000, help="Maximum frequency to use (in hz)")
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

    print("Loading corpus...")
    ycorpus = load_corpus(opt.corpus, sr=opt.sr, stereo=(opt.stereo==1))

    print("Finished setting up corpus; doing particle filter")
    pf = ParticleFilter(ycorpus=ycorpus, win=opt.winSize, sr=opt.sr, min_freq=opt.minFreq, max_freq=opt.maxFreq, p=opt.p, pfinal=opt.p, pd=opt.pd, temperature=opt.temperature, L=opt.L, P=opt.particles, gamma=opt.gamma, r=opt.r, neff_thresh=0.1*opt.particles, use_gpu=(opt.use_gpu==1), use_mic=(opt.target=="mic"))
    if not opt.target == "mic":
        ytarget = load_corpus(opt.target, sr=opt.sr, stereo=(opt.stereo==1))
        tic = time.time()
        pf.process_audio_offline(ytarget)
        print("Elapsed time particle filter: {:.3f}".format(time.time()-tic))    
    wavfile.write(opt.result, opt.sr, pf.get_generated_audio())
    

    if opt.saveplots == 1:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        pf.plot_statistics()
        plt.tight_layout()
        plt.savefig("{}.svg".format(opt.result), bbox_inches='tight')