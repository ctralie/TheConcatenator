"""
Programmer: Chris Tralie
Purpose: To serve as an entry point for my new musaicing technique
"""
import numpy as np
import argparse
import sys
sys.path.append("src")
from audioutils import load_corpus
from particle import ParticleFilter
import time
import torch
from scipy.io import wavfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help="Path to audio file or directory for source sounds")
    parser.add_argument('--target', type=str, required=True, help="Path to audio file for target sound, or \"mic\" if using the microphone to do real time")
    parser.add_argument('--result', type=str, required=True, help="Path to wav file to which to save the result")
    parser.add_argument('--recorded', type=str, default="recorded.wav", help="Path to wav file to which to the audio that was recorded (only relevant if target is mic)")
    parser.add_argument('--winSize', type=int, default=2048, help="Window Size in samples")
    parser.add_argument('--sr', type=int, default=44100, help="Sample rate")
    parser.add_argument('--minFreq', type=int, default=0, help="Minimum frequency to use (in hz), if using spectrogram bins directly")
    parser.add_argument('--maxFreq', type=int, default=8000, help="Maximum frequency to use (in hz), if using spectrogram bins directly")
    parser.add_argument('--useSTFT', type=int, default=1, help="If 1, use ordinary STFT bins")
    parser.add_argument('--useMel', type=int, default=0, help="If 1, use mel-spaced bins")
    parser.add_argument('--useZCS', type=int, default=0, help="If 1, use zero-crossings in each window as a feature")
    parser.add_argument('--melBands', type=int, default=40, help="Number of mel bands to use")
    parser.add_argument('--stereo', type=int, default=1, help="If 1, use stereo.  If 0, use mono")
    parser.add_argument('--device', type=str, default="cpu", help="Torch device to use")
    parser.add_argument("--nThreads", type=int, default=0, help="Use this number of threads in torch if specified")
    #parser.add_argument('--shiftrange', type=int, default=0, help="The number of halfsteps below and above which to shift the sound")
    parser.add_argument('--r', type=int, default=7, help="Width of the repeated activation filter")
    parser.add_argument('--p', type=int, default=10, help="Number of simultaneous activations to use in particle filter")
    parser.add_argument('--pFinal', type=int, default=0, help="Number of simultaneous activations to use in output (by default use all of them)")
    parser.add_argument('--pd', type=float, default=0.95, help="Probability of sticking to an activation (0 is no stick, closer to 1 is longer continuous activations)")
    parser.add_argument('--L', type=int, default=10, help="Number of KL iterations")
    parser.add_argument('--alpha', type=float, default=0.1, help="L2 penalty for activations")
    parser.add_argument('--particles', type=int, default=2000, help="Number of particles in the particle filter")
    parser.add_argument('--useTopParticle', type=int, default=0, help="If true, take activations only from the top particle.  Otherwise, aggregate them")
    parser.add_argument('--proposalK', type=int, default=0, help="Number of nearest neighbors to use in proposal distribution (if 0, don't use proposal distribution)")
    parser.add_argument('--temperature', type=float, default=50, help="Target importance.  Higher values mean activations will jump around more to match the target.")
    parser.add_argument('--saveplots', type=int, default=1, help='Save plots of iterations to disk')
    opt = parser.parse_args()

    if opt.nThreads > 0:
        print("Using {} threads".format(opt.nThreads))
        torch.set_num_threads(opt.nThreads)

    pfinal = opt.p
    if opt.pFinal > 0:
        pfinal = opt.pFinal
    assert(pfinal <= opt.p)

    print("Loading corpus audio...")
    tic = time.time()
    ycorpus = load_corpus(opt.corpus, sr=opt.sr, stereo=(opt.stereo==1))
    print("Finished loading up corpus audio: Elapsed Time {:.3f} seconds".format(time.time()-tic))

    feature_params = dict(
        win=opt.winSize,
        sr=opt.sr,
        min_freq=opt.minFreq,
        max_freq=opt.maxFreq,
        use_stft=opt.useSTFT == 1,
        use_mel=opt.useMel == 1,
        use_zcs=opt.useZCS == 1,
        mel_bands=opt.melBands
    )
    particle_params = dict(
        p=opt.p,
        pfinal=pfinal,
        pd=opt.pd,
        temperature=opt.temperature,
        L=opt.L,
        P=opt.particles,
        proposal_k=opt.proposalK,
        r=opt.r,
        neff_thresh=0.1*opt.particles,
        alpha=opt.alpha,
        use_top_particle=opt.useTopParticle == 1
    )
    pf = ParticleFilter(ycorpus, feature_params, particle_params, opt.device, opt.target=="mic")
    if opt.target == "mic":
        while not pf.recording_started or (pf.recording_started and not pf.recording_finished):
            time.sleep(2)
        recorded = pf.get_recorded_audio()
        wavfile.write(opt.recorded, opt.sr, recorded)
    else:
        print("Processing frames offline with particle filter...")
        ytarget = load_corpus(opt.target, sr=opt.sr, stereo=(opt.stereo==1))
        tic = time.time()
        pf.process_audio_offline(ytarget)
        print("Elapsed time offline particle filter: {:.3f}".format(time.time()-tic))    
    generated = pf.get_generated_audio()
    wavfile.write(opt.result, opt.sr, generated)

    print("\n\nMean frame time: {:.3f}ms\nRequired Upper Bound: {:.3f}ms\n".format(1000*np.mean(pf.frame_times), 500*opt.winSize/opt.sr))
    

    if opt.saveplots == 1:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        pf.plot_statistics()
        plt.tight_layout()
        plt.savefig("{}.svg".format(opt.result), bbox_inches='tight')
