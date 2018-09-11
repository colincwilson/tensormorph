#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr
from tpr import *

from scanner import BiScanner
from trimmer import BiTrimmer

# stem modifier (after Steriade, 1988)
class StemModifier(nn.Module):
    def __init__(self):
        super(StemModifier, self).__init__()
        self.deleter = BiScanner(morpho_size = tpr.dmorph+2, nfeature = 5, node = 'root-stem_modifier-deleter')
        self.trimmer = BiTrimmer(morpho_size = tpr.dmorph+2, nfeature = 5)
        self.delete_gate = Parameter(torch.zeros(1))
        self.trim_gate = Parameter(torch.zeros(1))

    def forward(self, stem, morpho):
        nbatch = stem.shape[0]     
        delete_gate = sigmoid(self.delete_gate)
        trim_gate = sigmoid(self.trim_gate)
        ones   = torch.ones((nbatch, tpr.nrole), requires_grad=False)

        delete = delete_gate * self.deleter(stem, morpho) +\
                 (1.0-delete_gate) * ones
        trim   = trim_gate * self.trimmer(stem, morpho) +\
                 (1.0-trim_gate) * ones
        copy = delete * trim
        return copy