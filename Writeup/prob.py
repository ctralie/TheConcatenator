import numpy as np

pd = 0.9
#N = 30000 # This is about how many grains are in the EdenVIP2 corpus. 
#N = 40000 # This is how many are in big-O
N = 1023999 # This is how many are in nsynth at 16000hz with a 512 window length
P = 10000
p = 5
delta = 2
w = 10

for k in range(1, 5):
    prob = 1-(pd + (1-pd)*(N-2-w*k)/N)**((2*delta+1)*p*P)
    print(k, prob)
