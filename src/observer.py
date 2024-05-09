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


class Observer:
    def __init__(self, p, W, WAlpha, L, temperature, device):
        """
        Constructor for a class that computes observation probabilities

        Parameters
        ----------
        p: int
            Number of activations
        W: torch.tensor(M, N)
            Templates matrix, assumed to sum to 1 down the columns
        WAlpha: torch.tensor(N)
            L2 penalty to put on activations from each column of W
        L: int
            Number of iterations of KL
        temperature: float
            Temperature for observation probabilities
        """
        self.p = p
        self.L = L
        self.temperature = temperature
        self.temperature_mutex = Lock()
        self.W = W
        self.WAlpha = WAlpha
        self.device = device
        if device == "np":
            WDenom = np.sum(W, axis=0)
        else:
            import torch
            WDenom = torch.sum(W, dim=0)
        WDenom[WDenom == 0] = 1
        self.WDenom = WDenom 
    
    def update_temperature(self, temperature):
        with self.temperature_mutex:
            self.temperature = temperature

    def observe_np(self, states, Vt):
        """
        Compute the observation probabilities for a set of states
        at a particular time.

        Parameters
        ----------
        states: ndarray(P, p)
            Column choices in W corresponding to each particle
        Vt: ndarray(M, 1)
            Observation for this time
        
        Returns
        -------
        ndarray(P)
            Observation probabilities
        """
        ## Step 1: Run NMF
        P = states.shape[0]
        p = states.shape[1]
        Wi = self.W[:, states]
        Wd = self.WDenom[states].reshape((P, p))
        hi = np.random.rand(P, p).astype(np.float32)
        Vt = np.reshape(Vt, (1, Vt.size))
        alpha = self.WAlpha[states].reshape((P, p))
        for _ in range(self.L):
            WH = np.einsum('jik,ik->ij', Wi, hi)
            WH[WH == 0] = 1
            VLam = Vt/WH
            hi *= np.einsum('jik,ij->ik', Wi, VLam)/(Wd + alpha*hi)
        ## Step 2: Compute KL divergences
        Vi = np.einsum('jik,ik->ij', Wi, hi)
        Vi[Vi == 0] = 1
        logarg = Vt/Vi
        logarg[logarg == 0] = 1
        kls = np.mean(Vt*np.log(logarg) - Vt + Vi, axis=1)
        ## Step 3: Compute observation probabilities
        with self.temperature_mutex:
            obs_prob = np.exp(-self.temperature*kls)
        denom = np.sum(obs_prob)
        if denom < 1e-40:
            obs_prob[:] = 1/obs_prob.size
        else:
            obs_prob /= denom
        return obs_prob

    def observe_torch(self, states, Vt):
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
        import torch
        ## Step 1: Run NMF
        P = states.shape[0]
        p = states.shape[1]
        Wi = self.W[:, states]
        Wi = torch.movedim(Wi, 1, 0)
        Wd = self.WDenom[states].view(P, p, 1)
        hi = torch.rand(P, p, 1).to(Wi)
        Vt = Vt.view(1, Vt.numel(), 1)
        alpha = self.WAlpha[states].view(P, p, 1)
        for _ in range(self.L):
            WH = torch.matmul(Wi, hi)
            WH[WH == 0] = 1
            VLam = Vt/WH
            hi *= torch.matmul(torch.movedim(Wi, 1, 2), VLam)/(Wd + alpha*hi)
        ## Step 2: Compute KL divergences
        Vi = torch.matmul(Wi, hi)
        Vi[Vi == 0] = 1
        logarg = Vt/Vi
        logarg[logarg == 0] = 1
        kls = torch.mean(Vt*torch.log(logarg) - Vt + Vi, dim=1)[:, 0]
        ## Step 3: Compute observation probabilities
        with self.temperature_mutex:
            obs_prob = torch.exp(-self.temperature*kls)
        denom = torch.sum(obs_prob)
        if denom < 1e-40:
            obs_prob[:] = 1/obs_prob.numel()
        else:
            obs_prob /= denom
        return obs_prob

    def observe(self, states, Vt):
        """
        Compute the observation probabilities for a set of states
        at a particular time.
        This is the fast GPU version

        Parameters
        ----------
        states: ndarray(P, p) or torch.tensor(P, p)
            Column choices in W corresponding to each particle
        Vt: ndarray(M, 1) or torch.tensor(M, 1)
            Observation for this time
        
        Returns
        -------
        ndarray(P) or torch.tensor(P)
            Observation probabilities
        """
        if self.device == "np":
            return self.observe_np(states, Vt)
        else:
            return self.observe_torch(states, Vt)
