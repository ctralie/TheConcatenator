## The Concatenator

This codebase has a basic python prototype of "The Concatenator," a fast, real time capable implementation of concatenative synthesis.

## Installation

### Installing Basic Requirements
Be sure to install numpy, matplotlib, scikit-learn, numba, librosa

~~~~~ bash
pip install numpy matplotlib scikit-learn numba librosa
~~~~~

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
