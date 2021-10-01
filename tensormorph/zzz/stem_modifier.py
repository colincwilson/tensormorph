#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
from tpr import *
from scanner import BiScanner
from trimmer import BiTrimmer

class StemModifier(nn.Module):
    """
    Stem modifier (after Steriade, 1988)
    """
    def __init__(self, dcontext = None):
        super(StemModifier, self).__init__()
        if dcontext is None:
            dcontext = config.dmorphosyn + 2
        self.deleter = BiScanner(dcontext = dcontext,
                                nfeature = 5,
                                npattern = 1)
        self.trimmer = BiTrimmer(dcontext = dcontext,
                                nfeature=5)
        self.delete_gate = Parameter(torch.randn(1))
        self.trim_gate = Parameter(torch.randn(1))

    def forward(self, stem, context):
        nbatch = stem.shape[0]     
        delete_gate = sigmoid(self.delete_gate).clamp(1.0,1.0) # xxx testing
        trim_gate = sigmoid(self.trim_gate) #.clamp(0.0,0.0)
        if config.discretize:
            delete_gate = torch.round(delete_gate)
            trim_gate = torch.round(trim_gate)

        delete = delete_gate * self.deleter(stem, context)
        trim = trim_gate * self.trimmer(stem, context)
        delete_or_trim = torch.cat(
            (delete.unsqueeze(2), trim.unsqueeze(2)), 2)
        # todo: replace max with log-space add followed by exp
        delete_or_trim, _ = torch.max(delete_or_trim, 2)
        copy = 1.0 - delete_or_trim  # note inversion!
        #copy = delete * trim

        # Mask out match results for epsilon fillers
        # todo: use tpr.epsilon_mask
        mask = hardtanh(stem.narrow(1,0,1), 0.0, 1.0)\
                .squeeze(1).detach()
        copy = copy * mask

        return copy