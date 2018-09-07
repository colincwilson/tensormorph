#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr
from tpr import *


# affix look-up table implemented with linear map from morphosyn + stem scan
class Thunker(nn.Module):
    def __init__(self, redup=False, root=True):
        super(Thunker, self).__init__()
        self.morph2affix =\
            nn.Linear(tpr.dmorph+2, tpr.dfill * tpr.drole, bias=True)
        self.morph2unpivot =\
            nn.Linear(tpr.dmorph+2, tpr.nrole, bias=True)

    def forward(self, morpho):
        nbatch = morpho.shape[0]
        # tpr of affix via binding matrix: affix tpr = F B_affix R^T
        #B_affix = self.morph2affix(morpho).view(nbatch, tpr.nfill, tpr.nrole)
        #B_affix = torch.exp(log_softmax(B_affix, dim=1)) # normalize within roles
        #affix = torch.bmm(torch.bmm(F, B_affix), Rt)
        # tpr of affix directly
        affix = self.morph2affix(morpho).view(nbatch, tpr.dfill, tpr.drole)
        affix = sigmoid(affix) # restrict learned affix components to [0,1]
        #affix = tanh(affix) # restrict learned affix components to [-1, +1]
        #affix  = tanh(PReLu(affix)) # restrict learned affix components to [0,1]
        #affix = bound_batch(affix)
        #affix = torch.zeros((nbatch, tpr.dfill, tpr.drole)) # xxx hack
        unpivot = self.morph2unpivot(morpho)
        unpivot = sigmoid(unpivot).view(nbatch, tpr.nrole)
        return affix, unpivot