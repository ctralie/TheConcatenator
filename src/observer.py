import numpy as np
import torch

class Observer:
    def __init__(self, p, W, L):
        """
        Constructor for a class that computes observation probabilities

        Parameters
        ----------
        p: int
            Number of activations
        W: torch.tensor(M, N)
            Templates matrix, assumed to sum to 1 down the columns
        L: int
            Number of iterations of KL
        """
        self.p = p
        self.W = W
        self.L = L

    def observe(self, states, Vt):
        """
        Compute the observation probabilities for a set of states
        at a particular time.
        This is the fast GPU version

        Parameters
        ----------
        states: torch.tensor(P, p)
            Column choices in W corresponding to each particle
        Vt: torch.tensor(M, 1)
            Observation for this time
        
        Returns
        -------
        torch.tensor(P)
            Observation probabilities
        """
        P = states.shape[0]
        p = states.shape[1]
        Wi = self.W[:, states]
        Wi = torch.movedim(Wi, 1, 0)
        Wd = torch.sum(Wi, dim=1).unsqueeze(-1)
        hi = torch.rand(P, p, 1).to(Wi)
        for _ in range(self.L):
            WH = torch.matmul(Wi, hi)
            WH[WH == 0] = 1
            VLam = Vt.unsqueeze(0)/WH
            hi *= torch.matmul(torch.movedim(Wi, 1, 2), VLam)/Wd
        Vi = torch.matmul(Wi, hi)
        ViNorms = torch.sqrt(torch.sum(Vi**2, dim=1, keepdims=True))
        ViNorms[ViNorms == 0] = 1
        Vi /= ViNorms
        return torch.sum(Vt.unsqueeze(0)*Vi, dim=1)[:, 0]