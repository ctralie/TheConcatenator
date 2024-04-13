import numpy as np

pd = 0.95
#N = 30000 # This is about how many grains are in the EdenVIP2 corpus. 
#N = 40000 # This is how many are in big-O
#N = 1023999 # This is how many are in nsynth at 16000hz with a 512 window length
P = 1000
N = 10000
p = 5
delta = 2
w = 11

for k in range(1, 5):
    prob = 1-(pd + (1-pd)*(N-1-w*k)/(N-1))**((2*delta+1)*p*P)
    print(k, prob)
