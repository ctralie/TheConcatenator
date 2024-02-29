import numpy as np
from scipy.io import wavfile

sr = 44100
f0 = 110
n_octaves = 4
x = np.array([])
for win in [int(0.5*sr), int(0.1*sr), int(0.05*sr)]:
    f = f0
    t = np.arange(win)/sr
    for i in range(12*n_octaves):
        x = np.concatenate((x, np.cos(2*np.pi*f*t)))
        f *= 2**(1/12)
x = np.array(x*32767, dtype=np.int16)
wavfile.write("sines.wav", sr, x)