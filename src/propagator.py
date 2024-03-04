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
        move_forward = (states < N-1)*(randPD < self.pd)
        states[move_forward == 1] += 1
        new_loc = ~move_forward
        n_new = torch.sum(new_loc)
        states[new_loc == 1] = torch.randint(N, size=(n_new,), dtype=torch.int32).to(self.device)

    ## TODO: Deal with resample colliding with same state (though this is negligible)
    def propagate_proposal(self, states, proposal):
        """
        Advance each particle forward randomly based on the transition model

        Parameters
        ----------
        states: torch.tensor(P, p, dtype=int32)
            Column choices in W corresponding to each particle.
            This is updated by reference
        proposal: torch.tensor(M)
            Indices to prioritize in proposal distribution

        Returns
        -------
        torch.tensor(P)
            The correction factor to apply to each particle
        """
        N = self.N
        ind = torch.zeros(N).to(self.device)
        ind[proposal] = 1
        other = torch.arange(N, dtype=torch.int32).to(self.device)
        other = other[ind == 0]

        p = torch.zeros(states.shape, dtype=torch.float32).to(self.device)
        q = torch.zeros(states.shape, dtype=torch.float32).to(self.device)

        randPD = torch.rand(states.shape).to(self.device)
        ## Continue on with probability pd
        move_forward = (states < N-1)*(randPD < self.pd)
        states[move_forward == 1] += 1
        p[move_forward] = self.pd
        q[move_forward] = self.pd
        new_loc = torch.ones(states.shape, dtype=torch.int32).to(self.device)
        new_loc[move_forward == 1] = 0
        p[new_loc] = (1-self.pd)/N
        
        ## Sample from proposal indices with probability (1-pd)/2
        new_loc[new_loc==1] *= (1+torch.randint(2, size=(torch.sum(new_loc),), dtype=torch.int32)).to(self.device)
        n_proposal = torch.sum(new_loc == 1)
        idxs = torch.randint(proposal.numel(), size=(n_proposal,), dtype=torch.int32).to(self.device)
        states[new_loc==1] = proposal[idxs]
        q[new_loc == 1] = (1-self.pd)/(2*proposal.numel())

        ## Sample from other indices with probability (1-pd)/2
        n_other = torch.sum(new_loc == 2)
        idxs = torch.randint(other.numel(), size=(n_other,), dtype=torch.int32).to(self.device)
        states[new_loc==2] = other[idxs]
        q[new_loc == 2] = (1-self.pd)/(2*other.numel())

        ## Correction factor
        p = torch.prod(p, dim=1)
        q = torch.prod(q, dim=1)
        q[q == 0] = 1
        p[q == 0] = 1
        return p/q