# -*- coding: utf-8 -*-

import config
from tpr import *


class MorphosynSequencer(nn.Module):
    """
    Sequence morphosyntactic specifications with inhibition
    """

    def __init__(self):  # xxx specify number of cogrammars
        super(MorphosynSequencer, self).__init__()
        # Dimension-dimension inhibition matrix
        if 0:
            W_ = torch.ones(config.ndim, config.ndim)
            W1_ = W_.triu() * 1.0
            W2_ = W_.tril() * -10.0
            W_ = (W1_ + W2_)
            W_.diagonal().fill_(-10.0)
            self.W = W_
        else:
            self.W = Parameter(0.1 * torch.randn(config.ndim, config.ndim))
            self.W.diagonal().clamp(-10.0, -10.0)  # No self-inhibition

    def forward(self, morphosyn, nslot):
        # Determine dimensions that are specified
        alpha = [morphosyn[i][:, 0] == 0.0 for i in range(config.ndim)]
        alpha = torch.stack(alpha, -1).float()

        # Inhibitory strengths
        W = exp(self.W)

        # Apply inhibition and suppression in each slot
        gammas = []
        for i in range(nslot):
            beta = einsum('bj,ij->bi', alpha, W)
            gamma = alpha * exp(-beta)
            alpha = alpha * (1.0 - gamma)
            gammas.append(gamma)

        Mslot2dim_attn = torch.stack(gammas, -1)
        #print(slot2dim_attn.shape)
        #sys.exit(0)
        return Mslot2dim_attn