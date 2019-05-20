#!/usr/bin/env python
# -*- coding: utf-8 -*-


from .environ import config
from .tpr import *
from .scanner import BiScanner
from .trimmer import BiTrimmer

# stem modifier (after Steriade, 1988)
class StemModifier(nn.Module):
    def __init__(self):
        super(StemModifier, self).__init__()
        self.deleter = BiScanner(morpho_size=(config.dmorph+2), nfeature=5, npattern=1, node='root-stem_modifier-deleter')
        self.trimmer = BiTrimmer(morpho_size=(config.dmorph+2), nfeature=5)
        self.delete_gate = Parameter(torch.zeros(1))
        self.trim_gate = Parameter(torch.zeros(1))

    def forward(self, stem, morpho):
        nbatch = stem.shape[0]     
        delete_gate = sigmoid(self.delete_gate).clamp(1.0,1.0)
        trim_gate = sigmoid(self.trim_gate).clamp(0.0,0.0)
        if config.discretize:
            delete_gate = torch.round(delete_gate)
            trim_gate = torch.round(trim_gate)

        delete = delete_gate * self.deleter(stem, morpho)
        trim = trim_gate * self.trimmer(stem, morpho)
        delete_or_trim = torch.cat(
            (delete.unsqueeze(2), trim.unsqueeze(2)), 2)
        delete_or_trim, _ = torch.max(delete_or_trim, 2)
        copy    = 1.0 - delete_or_trim
        #copy = delete * trim

        # Mask out match results for epsilon fillers
        #mask = hardtanh(stem.narrow(1,0,1), 0.0, 1.0)\
        #        .squeeze(1).detach()
        #copy = copy * mask

        return copy