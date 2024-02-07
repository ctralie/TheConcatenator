import numpy as np

from numba import jit

@jit(nopython=True)
def propagate_numba_helper(states, randPD, randState, N, pd):
    idx = 0
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            if states[i][j] < N-1 and randPD[idx] < pd:
                states[i][j] += 1
            else:
                stateOut = randState[idx]
                if stateOut == states[i][j]:
                    stateOut = N-2
                if stateOut == states[i][j]+1:
                    stateOut = N-1
                states[i][j] = stateOut
            idx += 1

class Propagator:
    def __init__(self, N, pd):
        """
        Constructor for a class that computes transition probabilities
        quickly using moderngl

        Parameters
        ----------
        N: int
            Number of corpus states
        pd: float
            Probability of remaining in the same column in time order
        """
        self.N = N
        self.pd = pd

    def propagate_numba(self, states):
        """
        Compute the observation probabilities for a set of states
        at a particular time.
        This is a CPU numba version, which is faster than the CPU version
        but slower than the GPU version

        Parameters
        ----------
        states: ndarray(P, p)
            Column choices in W corresponding to each particle.
            This is updated by reference
        """
        randPD = np.random.rand(states.size)
        randState = np.random.randint(self.N-2, size=states.size)
        propagate_numba_helper(states, randPD, randState, self.N, self.pd)


    def propagate_cpu(self, states, proposal_idxs=[], rejection_sample=False):
        """
        For each particle, sample from the proposal distribution

        Parameters
        ----------
        states: ndarray(P, p)
            Column choices in W corresponding to each particle.
            This is updated by reference
        proposal_idxs: ndarray(<= N)
            Subset of indices of W to choose more frequently
            based on observations
        rejection_sample: bool
            Whether to do rejection sampling to prevent collisions of activations
        
        Returns
        -------
        ndarray(P, p): The final states
        """
        P = states.shape[0]
        p = states.shape[1]

        state_next = np.zeros(p, dtype=int)
        for i in range(P):
            ## Step 1: Sample from proposal distribution for each particle
            finished = False
            while not finished: # Do rejection sampling
                for j in range(p):
                    if states[i][j] < self.N-1 and np.random.rand() < self.pd:
                        state_next[j] = states[i][j] + 1
                    else:
                        if len(proposal_idxs) == 0 or np.random.rand() < 0.25:
                            # Choose a random element not equal to state[j] or state[j]+1
                            next = np.random.randint(self.N-2)
                        else:
                            # Choose a random element from the proposal set
                            next = np.random.choice(proposal_idxs)
                        # Ex) N = 10, state[j] = 6; avoid 6 and 7
                        if next == states[i][j]: # next = 6, make next 8
                            next = self.N-2
                        if next == states[i][j]+1: # next 7, make next 9
                            next = self.N-1
                        state_next[j] = next
                if not rejection_sample or (len(np.unique(state_next)) == p):
                    finished = True
                    states[i, :] = state_next