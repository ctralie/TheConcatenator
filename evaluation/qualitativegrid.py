import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import os
import time
import subprocess
import pickle
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
    pd=0.95,
    temperature=10,
    L=10,
    P=P,
    r=3,
    neff_thresh=0.1*P,
    proposal_k=0,
    alpha=0.1,
    use_top_particle=False
)

resultsdir = "Qualitative/Results/P{}_p{}_temp{}_pd{}".format(particle_params["P"], particle_params["p"], particle_params["temperature"], particle_params["pd"])
if not os.path.exists(resultsdir):
    print("Making", resultsdir)
    os.mkdir(resultsdir)
pickle.dump(particle_params, open("{}/particle_params.pkl".format(resultsdir), "wb"))
pickle.dump(feature_params, open("{}/feature_params.pkl".format(resultsdir), "wb"))

targets = [
            "2 Voice Harmony/Let It Be - 2 voice cp - filt saw - 2 octaves.wav",
            "2 Voice Harmony/Let It Be - 2 voice cp - sine - 2 octaves.wav",
            "3 Voice Harmony/3 voice cp - sine - 2 oct, 1st.wav",
            "3 Voice Harmony/3 voice cp - synth filt - 2 oct, 1st.wav",
            "Melody/Let It Be - Melody - filt saw - 2 octaves.wav",
            "Melody/Let It Be - Melody - sine - 2 octaves.wav",
            "Ripple Continuum/Ripple Contiuum - filt saw - 3 octaves.wav",
            "Ripple Continuum/Ripple Contiuum Piano Solo.wav",
            "Ripple Continuum/Ripple Contiuum - sine - 3 octaves.wav",
            "Bass Test.wav",
            "Beatbox Test.wav",
            "Drums Bass Test.wav",
            "Pink Noise Sweet Test.wav",
            "Skrillex - Scary Monsters and Nice Sprites.m4a",
            "Vocal Test.wav",
            "Beatles_LetItBe.mp3"
           ]

corpora = [
    "Bees_Buzzing.mp3",
    "Beethoven - Symphony No. 5 in C Minor, Op. 67_ I. Allegro con brio.m4a",
    "Bjork It's Oh So Quiet.m4a",
    "Skrillex - Scary Monsters and Nice Sprites.m4a",
    "Foley FX",
    "Vocals",
    "Percussive",
    "Pink Floyd - The Wall",
    "Skrillex - Quest For Fire",
    "Mr. Bill - Spectra Sample Pack Excerpt",
    "Edenic Mosaics LIB Compare",
    "Woodwinds"
]

## Step 2: Initialize corpus
html = "<html>\n<body>\n<table>\n<tr><td></td>"
for target in targets:
    html += "<td><h3>{}</h3></td>".format(target)
html += "</tr>"
for corpus in corpora:
    corpus_name = corpus.split("/")[-1]
    ycorpus = load_corpus("Qualitative/Corpus/" + corpus, sr, stereo=True)
    pf = ParticleFilter(ycorpus, feature_params, particle_params, 'cuda')
    html += "<tr><td><h3>{}</h3></td>".format(corpus)
    for target in targets:
        target_name = target.split("/")[-1]
        print(corpus, target)
        path = "{}_{}.wav".format(corpus_name, target_name)
        pathmp3 = "{}_{}.mp3".format(corpus_name, target_name)
        if not os.path.exists(resultsdir+os.path.sep+pathmp3):
            pf.reset_state()
            ytarget = load_corpus("Qualitative/Targets/"+target, sr=sr, stereo=stereo)
            tic = time.time()
            pf.process_audio_offline(ytarget)
            generated = pf.get_generated_audio()
            wavfile.write(path, sr, generated)
            if os.path.exists(resultsdir+os.path.sep+pathmp3):
                os.remove(resultsdir+os.path.sep+pathmp3)
            cmd = ["ffmpeg", "-i", path, resultsdir+os.path.sep+pathmp3]
            print(cmd)
            subprocess.call(cmd)
            os.remove(path)
        html += "<td><audio controls><source src=\"{}\" type=\"audio/mp3\"></audio></td>".format(pathmp3)
        fout = open("{}/index.html".format(resultsdir), "w")
        fout.write(html)
        fout.close()
    html += "</tr>\n"

