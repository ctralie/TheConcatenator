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
couple_channels = False
P = 1000
p = 5
pd = 0.95
temperature = 10
feature_params = dict(
    win=win,
    sr=sr,
    min_freq=0,
    max_freq=8000,
    use_stft=True,
    use_mel=False,
    mel_bands=40,
)
particle_params = dict(
    p=p,
    pfinal=p,
    pd=pd,
    temperature=temperature,
    L=10,
    P=P,
    r=3,
    neff_thresh=0.1*P,
    proposal_k=0,
    alpha=0.1,
    use_top_particle=False
)

resultsdir = "Qualitative/Results/P{}_p{}_temp{}_pd{}_win{}".format(particle_params["P"], particle_params["p"], particle_params["temperature"], particle_params["pd"], win)
if not os.path.exists(resultsdir):
    print("Making", resultsdir)
    os.mkdir(resultsdir)
pickle.dump(particle_params, open("{}/particle_params.pkl".format(resultsdir), "wb"))
pickle.dump(feature_params, open("{}/feature_params.pkl".format(resultsdir), "wb"))

all_targets = dict(
    inTheWild = [
        "Beatles_LetItBe.mp3",
        "Skrillex - Scary Monsters and Nice Sprites.m4a",
        "Pink Noise Sweet Test.wav",
        "Vocal Test.wav",
        "Bass Test.wav",
        "Beatbox Test.wav",
        "Drums Bass Test.wav"
    ],
    syntheticTunes = [
        "2 Voice Harmony/Let It Be - 2 voice cp - filt saw - 2 octaves.wav",
        "2 Voice Harmony/Let It Be - 2 voice cp - sine - 2 octaves.wav",
        "3 Voice Harmony/3 voice cp - sine - 2 oct, 1st.wav",
        "3 Voice Harmony/3 voice cp - synth filt - 2 oct, 1st.wav",
        "Melody/Let It Be - Melody - filt saw - 2 octaves.wav",
        "Melody/Let It Be - Melody - sine - 2 octaves.wav",
        "Ripple Continuum/Ripple Contiuum - filt saw - 3 octaves.wav",
        "Ripple Continuum/Ripple Contiuum Piano Solo.wav",
        "Ripple Continuum/Ripple Contiuum - sine - 3 octaves.wav"
    ]
)

corpora = [
    dict(
        path="Bees_Buzzing.mp3",
        html="""
            Driedger's Bees 
            <audio controls>
                <source src="../Corpus/Bees_Buzzing.mp3">
            </audio>
        """
    ),
    dict(
        path="Beethoven - Symphony No. 5 in C Minor, Op. 67_ I. Allegro con brio.m4a",
        html="""
            <a href = "https://www.youtube.com/watch?v=4aAne1Ol7R0">Beethoven - Symphony No. 5 in C Minor, Op. 67_ I. Allegro con brio</a>
        """
    ),
    dict(
        path="Bjork It's Oh So Quiet.m4a",
        html="""
            <a href = "https://www.youtube.com/watch?v=HMt-nAYKTN0">Bjork It's Oh So Quiet</a>
        """
    ),
    dict(
        path="Skrillex - Scary Monsters and Nice Sprites.m4a",
        html="""
            Skrillex - Scary Monsters and Nice Sprites 
            <audio controls>
                <source src="../Targets/Skrillex - Scary Monsters and Nice Sprites.m4a">
            </audio>
        """
    ),
    dict(
        path="Foley FX",
        html="Foley FX"
    ),
    dict(
        path="Vocals",
        html="Vocals"
    ),
    dict(
        path="Percussive",
        html="Percussive"
    ),
    dict(
        path="Pink Floyd - The Wall",
        html="""
            <a href = "https://www.youtube.com/watch?v=iLFwTqdsuxw&list=PLyDzU3p8FP24syYfTXpGqTDHsQhlxwllS">Pink Floyd - The Wall (Full Album)</a>
        """
    ),
    dict(
        path="Skrillex - Quest For Fire",
        html="""
            <a href = "https://www.youtube.com/playlist?list=OLAK5uy_mWpqMuJUmJT8gnimi-1ZriwVdgFyMLems">Skrillex - Quest for Fire (Full Album)</a>
        """
    ),
    dict(
        path="Mr. Bill - Spectra Sample Pack Excerpt",
        html="""
            <a href = "https://mrbillstunes.com/">Mr. Bill</a> - Spectra Sample Pack Excerpt
        """
    ),
    dict(
        path="Edenic Mosaics LIB Compare",
        html="Edenic Mosaics LIB Compare"
    ),
    dict(
        path="Woodwinds",
        html="""
            All woodwinds from the <a href = "https://theremin.music.uiowa.edu/">Pre-2012 UIowa Dataset</a>
        """
    )
]

## Step 2: Initialize corpus
for targets_name, targets in all_targets.items():
    html = """
        <html>
        <style>
        #results {
        font-family: Arial, Helvetica, sans-serif;
        border-collapse: collapse;
        width: 100%;
        }

        #results td, #results th {
        border: 1px solid #ddd;
        padding: 8px;
        }

        #results tr:nth-child(even){background-color: #f2f2f2;}

        #results tr:hover {background-color: #ddd;}

        #results th {
            padding-top: 12px;
            padding-bottom: 12px;
            text-align: left;
            background-color: #04AA6D;
            color: white;
        }
        </style>
        <body>
    """
    html += "<h2>{}</h2>".format(targets_name)
    html += f"<h3>sr={sr}, win={win}, P={P}, p={p}, p<SUB>d</SUB>={pd}, &tau;={temperature}</h3>"
    html += "<table id=\"results\">\n<tr><td></td>"
    for target in targets:
        html += "<th><h3>{}</h3><audio controls><source src=\"../Targets/{}.m4a\"></audio></th>".format(
            target.split("/")[-1][0:-4],
            target[0:-4]
            )
    html += "</tr>\n\n"
    for corpus in corpora:
        corpus, corpus_html = corpus["path"], corpus["html"]
        corpus_name = corpus.split("/")[-1]
        ycorpus = load_corpus("Qualitative/Corpus/" + corpus, sr, stereo=stereo)
        pf = ParticleAudioProcessor(ycorpus, feature_params, particle_params, 'cuda', couple_channels)
        html += "\n\n<tr>\n    <td><h3>{}</h3></td>".format(corpus_html)
        for target in targets:
            target_name = target.split("/")[-1]
            print(corpus, target)
            path = "{}_{}.wav".format(corpus_name, target_name)
            pathm4a = "{}_{}.m4a".format(corpus_name, target_name)
            if not os.path.exists(resultsdir+os.path.sep+pathm4a):
                pf.reset_state()
                ytarget = load_corpus("Qualitative/Targets/"+target, sr=sr, stereo=stereo)
                tic = time.time()
                pf.process_audio_offline(ytarget)
                generated = pf.get_generated_audio()
                wavfile.write(path, sr, generated)
                if os.path.exists(resultsdir+os.path.sep+pathm4a):
                    os.remove(resultsdir+os.path.sep+pathm4a)
                cmd = ["ffmpeg", "-i", path, resultsdir+os.path.sep+pathm4a]
                print(cmd)
                subprocess.call(cmd)
                os.remove(path)
            html += "\n    <td><audio controls><source src=\"{}\"></audio></td>".format(pathm4a)
            fout = open("{}/{}.html".format(resultsdir, targets_name), "w")
            fout.write(html)
            fout.close()
        html += "</tr>\n"
    html += "</table>\n</body>\n</html>"
    fout = open("{}/{}.html".format(resultsdir, targets_name), "w")
    fout.write(html)
    fout.close()

