## The Concatenator

This codebase has a basic python prototype of "The Concatenator," a fast, real time capable implementation of concatenative synthesis.

## Installation

### Installing Basic Requirements
If you do not have python on your system, follow the instructions at <a href = "https://docs.anaconda.com/free/miniconda/">this link</a> to download and install miniconda.  Then, open up the anaconda prompt as indicated at that link, and change directories into <code>TheConcatenator</code> folder.

Then, regardless of whether this is a fresh install or an old install, make sure you have the following dependencies installed: numpy, matplotlib, scikit-learn, numba, librosa

~~~~~ bash
pip install numpy matplotlib scikit-learn numba librosa
~~~~~

It is also highly recommended that you install cuda and torch if you have a cuda-capable Nvidia GPU.  As of the writing of this readme (4/29/2024), the commands to install the best versions of this in miniconda are the following, in sequence:

~~~~~ bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
~~~~~

~~~~~ bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
~~~~~

Beware that this will download ~10GB of data to your computer, and so it may take a moment.


### Installation for real time
If you want to do real time, be sure that portaudio is installed.  On mac, for instance

~~~~~ bash
brew install portaudio
~~~~~

Then, you can install pyaudio

~~~~~ bash
pip install pyaudio
~~~~~

## Example Usage
Type

~~~~~ bash
python musaic.py --help
~~~~~


for all options.  For instance, to create the "Let It Bee" example offline, you can run

~~~~~ bash
python musaic.py --corpus corpus/Bees_Buzzing.mp3 --target target/Beatles_LetItBe.mp3 --minFreq 0 --maxFreq 8000 --particles 1000 --pd 0.95 --temperature 50 --p 5 --device np --result 1000Particles.wav
~~~~~

The above works best on a mac.  But if you're on windows and linux and you have pytorch installed with cuda support, you can run the following instead, which will go much faster by using the GPU

~~~~~ bash
python musaic.py --corpus corpus/Bees_Buzzing.mp3 --target target/Beatles_LetItBe.mp3 --minFreq 0 --maxFreq 8000 --particles 1000 --pd 0.95 --temperature 50 --p 5 --device cuda --result 1000Particles.wav
~~~~~

You can also use torch with the cpu, which sometimes threads better than numpy
~~~~~ bash
python musaic.py --corpus corpus/Bees_Buzzing.mp3 --target target/Beatles_LetItBe.mp3 --minFreq 0 --maxFreq 8000 --particles 1000 --pd 0.95 --temperature 50 --result 1000Particles.wav --p 5 --device cpu
~~~~~

Then, based on whichever device is working the fastest, you can launch a real time session by changing --target to "mic".  It also helps to switch to mono instead of stereo so the code goes faster.  For instance, suppose cuda works for you.  Then you can say
~~~~~ bash
python musaic.py --corpus corpus/Bees_Buzzing.mp3 --target mic --minFreq 0 --maxFreq 8000 --particles 1000 --pd 0.95 --temperature 50 --p 5 --device cuda --stereo 0 --result 1000Particles.wav
~~~~~

Depending on your system, you may be able to use more particles and more activations (the --p parameter).  But if you get "underrun occurred" regularly in the console, it means that the program can't keep up with the parameters you've chosen in the real time scenario.
