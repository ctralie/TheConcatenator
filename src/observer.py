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
        self.L = L
        # Normalize ahead of time
        WDenom = torch.sum(W, dim=0, keepdims=True)
        WDenom[WDenom == 0] = 1
        self.W = W/WDenom 

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
        hi = torch.rand(P, p, 1).to(Wi)
        Vt = Vt.view(1, Vt.numel(), 1)
        for _ in range(self.L):
            WH = torch.matmul(Wi, hi)
            WH[WH == 0] = 1
            VLam = Vt/WH
            hi *= torch.matmul(torch.movedim(Wi, 1, 2), VLam)
        Vi = torch.matmul(Wi, hi)
        return torch.mean(Vt*torch.log(Vt/Vi) - Vt + Vi, dim=1)[:, 0]