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

class Propagator:
    def __init__(self, corpus_labels, pd, pr, device):
        """
        Constructor for a class that computes transition probabilities

        Parameters
        ----------
        corpus_labels: ndarray(N, dtype=int)
            File label of each corpus element
        pd: float
            Probability of remaining in the same column in time order
        pr: float
            Probability that an activation will be reversed
        device: str
            Device on which to do the computation
        """
        self.N = len(corpus_labels)
        corpus_labels = np.concatenate((corpus_labels, -np.ones(2))) # Dummy to deal with boundaries 
        if device != "np":
            import torch
            corpus_labels = torch.from_numpy(corpus_labels).to(device)
        self.corpus_labels = corpus_labels
        self.pd = pd
        self.pr = pr
        self.device = device

    def update_pd(self, pd):
        with self.pd_mutex:
            self.pd = pd

    def get_avg_activation_len(self):
        """
        Compute the average activation length according to the negative binomial 
        distribution
        """
        return self.pd/(1-self.pd)

    def propagate(self, states, forward):
        """
        Advance each particle forward randomly based on the transition model
        NOTE: For ease of implementation, the probability of remaining fixed
        is technically p + 1/N, but that should be very close to p

        Parameters
        ----------
        states: torch.tensor(P, p, dtype=int32)
            Column choices in W corresponding to each particle.
            This is updated by reference
        forward: torch.tensor(P, p, dtype=int32)
            Indicator as to whether each grain is moving forward(1) or in reverse(0)
            This is updated by reference
        """
        pd = None
        pd = self.pd
        N = self.N
        labels = self.corpus_labels
        if self.device == "np":
            randPD = np.random.rand(*states.shape)
        else:
            import torch
            randPD = torch.rand(states.shape).to(self.device)
        # Deal with all activations moving forward
        keep_moving_forward = forward*(states < N-1)*(randPD < pd)*(labels[states+1] == labels[states])
        states[keep_moving_forward == 1] += 1
        # Deal with all activations moving in reverse
        keep_moving_backward = (forward==0)*(states > 0)*(randPD < pd)*(labels[states-1] == labels[states])
        states[keep_moving_backward == 1] -= 1
        # Determine all states that need to be resampled and perform the resampling
        new_loc = (keep_moving_forward == 0)*(keep_moving_backward == 0)
        if self.device == "np":
            n_new = np.sum(new_loc)
            states[new_loc == 1] = np.random.randint(N, size=(n_new,))
            if self.pr > 0 and self.pr < 1:
                forward[new_loc == 1] = np.random.rand(n_new) > self.pr
        else:
            import torch
            n_new = torch.sum(new_loc)
            states[new_loc == 1] = torch.randint(N, size=(n_new,), dtype=torch.int32).to(self.device)
            if self.pr > 0 and self.pr < 1:
                forward[new_loc == 1] = (torch.rand(n_new) > self.pr).int().to(self.device)