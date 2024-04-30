"""
Code copyright Christopher J. Tralie, 2024
Attribution-NonCommercial-ShareAlike 4.0 International


Share — copy and redistribute the material in any medium or format
The licensor cannot revoke these freedoms as long as you follow the license terms.

 Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made . You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    NonCommercial — You may not use the material for commercial purposes .
    NoDerivatives — If you remix, transform, or build upon the material, you may not distribute the modified material.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
"""

import numpy as np
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
        NOTE: For ease of implementation, the probability of remaining fixed
        is technically p + 1/N, but that should be very close to p

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
        if self.device == "np":
            randPD = np.random.rand(*states.shape)
        else:
            import torch
            randPD = torch.rand(states.shape).to(self.device)
        move_forward = (states < N-1)*(randPD < pd)
        states[move_forward == 1] += 1
        new_loc = ~move_forward
        if self.device == "np":
            n_new = np.sum(new_loc)
            states[new_loc == 1] = np.random.randint(N, size=(n_new,))
        else:
            import torch
            n_new = torch.sum(new_loc)
            states[new_loc == 1] = torch.randint(N, size=(n_new,), dtype=torch.int32).to(self.device)
        
    def propagate_proposal_np(self, states, proposal, v=2):
        """
        Advance each particle forward randomly based on the transition model

        Parameters
        ----------
        states: ndarray(P, p)
            Column choices in W corresponding to each particle.
            This is updated by reference
        proposal: ndarray(M)
            Indices to prioritize in proposal distribution
        v: int
            Choose proposal indices (v-1)/v of the time when a jump happens

        Returns
        -------
        ndarray(P)
            The correction factor to apply to each particle
        """
        pd = None
        with self.pd_mutex:
            pd = self.pd
        N = self.N
        ind = np.zeros(N)
        ind[proposal] = 1
        other = np.arange(N)
        other = other[ind == 0]

        p = np.zeros(states.shape, dtype=np.float32)
        q = np.zeros(states.shape, dtype=np.float32)

        randPD = np.random.rand(*states.shape)
        ## Continue on with probability pd
        move_forward = (states < N-1)*(randPD < pd)
        states[move_forward == 1] += 1
        p[move_forward] = pd
        p[~move_forward] = (1-pd)/N
        q[move_forward] = pd
        new_loc = np.ones(states.shape, dtype=np.int32)
        new_loc[move_forward == 1] = 0
        
        ## Sample from proposal indices with probability (1-pd)(v-1)/v
        new_loc[new_loc==1] = 1+np.random.randint(v, size=(np.sum(new_loc),))
        n_proposal = np.sum(new_loc > 1)
        idxs = np.random.randint(proposal.size, size=(n_proposal,))
        q[new_loc > 1] = ((v-1)/v)*(1-pd)/proposal.size
        # If we happen to jump to the next state, be sure to incorporate this probability properly
        q[new_loc > 1][states[new_loc > 1] + 1 == proposal[idxs]] += pd
        p[new_loc > 1][states[new_loc > 1] + 1 == proposal[idxs]] += pd
        states[new_loc > 1] = proposal[idxs]

        ## Sample from other indices with probability (1-pd)/v
        n_other = np.sum(new_loc == 1)
        idxs = np.random.randint(other.size, size=(n_other,))
        q[new_loc == 1] = (1/v)*(1-pd)/other.size
        # If we happen to jump to the next state, be sure to incorporate this probability properly
        q[new_loc == 1][states[new_loc == 1] + 1 == other[idxs]] += pd
        p[new_loc == 1][states[new_loc == 1] + 1 == other[idxs]] += pd
        states[new_loc==1] = other[idxs]

        ## Correction factor
        num = np.prod(p, axis=1)
        denom = np.prod(q, axis=1)
        num[denom == 0] = 1
        denom[denom == 0] = 1
        return num/denom

    def propagate_proposal_torch(self, states, proposal, v=2):
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
        import torch
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

    def propagate_proposal(self, states, proposal, v=2):
        """
        Advance each particle forward randomly based on the transition model

        Parameters
        ----------
        states: ndarray(P, p) or torch.tensor(P, p, dtype=int32)
            Column choices in W corresponding to each particle.
            This is updated by reference
        proposal: ndarray(M) or torch.tensor(M)
            Indices to prioritize in proposal distribution
        v: int
            Choose proposal indices (v-1)/v of the time when a jump happens

        Returns
        -------
        ndarray(P) or torch.tensor(P)
            The correction factor to apply to each particle
        """
        if self.device == "np":
            return self.propagate_proposal_np(states, proposal, v)
        else:
            return self.propagate_proposal_torch(states, proposal, v)