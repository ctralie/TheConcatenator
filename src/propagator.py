import numpy as np
import torch
from threading import Lock

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
        self.pd_mutex = Lock()

    def update_pd(self, pd):
        with self.pd_mutex:
            self.pd = pd

    def get_avg_activation_len(self):
        """
        Compute the average activation length according to the negative binomial 
        distribution
        """
        ret = 1
        with self.pd_mutex:
            ret = self.pd/(1-self.pd)
        return ret

    def propagate(self, states):
        """
        Advance each particle forward randomly based on the transition model

        Parameters
        ----------
        states: torch.tensor(P, p, dtype=int32)
            Column choices in W corresponding to each particle.
            This is updated by reference
        """
        pd = None
        with self.pd_mutex:
            pd = self.pd
        N = self.N
        randPD = torch.rand(states.shape).to(self.device)
        move_forward = (states < N-1)*(randPD < pd)
        states[move_forward == 1] += 1
        new_loc = ~move_forward
        n_new = torch.sum(new_loc)
        states[new_loc == 1] = torch.randint(N, size=(n_new,), dtype=torch.int32).to(self.device)
        
    def propagate_proposal(self, states, proposal, v=2):
        """
        Advance each particle forward randomly based on the transition model

        Parameters
        ----------
        states: torch.tensor(P, p, dtype=int32)
            Column choices in W corresponding to each particle.
            This is updated by reference
        proposal: torch.tensor(M)
            Indices to prioritize in proposal distribution
        v: int
            Choose proposal indices (v-1)/v of the time when a jump happens

        Returns
        -------
        torch.tensor(P)
            The correction factor to apply to each particle
        """
        pd = None
        with self.pd_mutex:
            pd = self.pd
        N = self.N
        ind = torch.zeros(N).to(self.device)
        ind[proposal] = 1
        other = torch.arange(N, dtype=torch.int32).to(self.device)
        other = other[ind == 0]

        p = torch.zeros(states.shape, dtype=torch.float32).to(self.device)
        q = torch.zeros(states.shape, dtype=torch.float32).to(self.device)

        randPD = torch.rand(states.shape).to(self.device)
        ## Continue on with probability pd
        move_forward = (states < N-1)*(randPD < pd)
        states[move_forward == 1] += 1
        p[move_forward] = pd
        p[~move_forward] = (1-pd)/N
        q[move_forward] = pd
        new_loc = torch.ones(states.shape, dtype=torch.int32).to(self.device)
        new_loc[move_forward == 1] = 0
        
        ## Sample from proposal indices with probability (1-pd)(v-1)/v
        new_loc[new_loc==1] = (1+torch.randint(v, size=(torch.sum(new_loc),), dtype=torch.int32)).to(self.device)
        n_proposal = torch.sum(new_loc > 1)
        idxs = torch.randint(proposal.numel(), size=(n_proposal,), dtype=torch.int32).to(self.device)
        q[new_loc > 1] = ((v-1)/v)*(1-pd)/proposal.numel()
        # If we happen to jump to the next state, be sure to incorporate this probability properly
        q[new_loc > 1][states[new_loc > 1] + 1 == proposal[idxs]] += pd
        p[new_loc > 1][states[new_loc > 1] + 1 == proposal[idxs]] += pd
        states[new_loc > 1] = proposal[idxs]

        ## Sample from other indices with probability (1-pd)/v
        n_other = torch.sum(new_loc == 1)
        idxs = torch.randint(other.numel(), size=(n_other,), dtype=torch.int32).to(self.device)
        q[new_loc == 1] = (1/v)*(1-pd)/other.numel()
        # If we happen to jump to the next state, be sure to incorporate this probability properly
        q[new_loc == 1][states[new_loc == 1] + 1 == other[idxs]] += pd
        p[new_loc == 1][states[new_loc == 1] + 1 == other[idxs]] += pd
        states[new_loc==1] = other[idxs]

        ## Correction factor
        num = torch.prod(p, dim=1)
        denom = torch.prod(q, dim=1)
        num[denom == 0] = 1
        denom[denom == 0] = 1
        return num/denom