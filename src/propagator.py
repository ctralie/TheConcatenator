import numpy as np
import torch

class Propagator:
    def __init__(self, N, pd, device):
        """
        Constructor for a class that computes transition probabilities

        Parameters
        ----------
        N: int
            Number of corpus states
        pd: float
            Probability of remaining in the same column in time order
        """
        self.N = N
        self.pd = pd
        self.device = device

    def propagate(self, states):
        """
        Advance each particle forward randomly based on the transition model

        Parameters
        ----------
        states: torch.tensor(P, p, dtype=int32)
            Column choices in W corresponding to each particle.
            This is updated by reference
        """
        N = self.N
        randPD = torch.rand(states.shape).to(self.device)
        randState = torch.randint(N-2, size=states.shape, dtype=torch.int32).to(self.device)
        move_forward = (states < N-1)*(randPD < self.pd)
        states[move_forward == 1] += 1
        new_loc = ~move_forward
        stateOut = randState[new_loc == 1]
        # If we hit the same state as before, sample N-2
        stateOut[stateOut == states[new_loc == 1]] = N-2
        # If we hit the state after the last state, sample N-1
        stateOut[stateOut == states[new_loc == 1]+1] = N-1
        states[new_loc == 1] = stateOut
