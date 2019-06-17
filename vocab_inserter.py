#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
from .tpr import *


class VocabInserter(nn.Module):
    """
    Vocabulary item look-up table implemented with linear map from morphosyn + stem scan
    """
    def __init__(self, redup=False, root=True):
        super(VocabInserter, self).__init__()
        self.morph2affix =\
            nn.Linear(config.dmorph+2, config.dfill * config.drole, bias=True)
        self.morph2unpivot =\
            nn.Linear(config.dmorph+2, config.nrole, bias=True)
        self.morph2copy =\
            nn.Linear(config.dmorph+2, config.nrole, bias=True)
        #self.morph2affix.bias.data.fill_(-2.5)
        #self.morph2unpivot.bias.data.fill_(2.5)

    def forward(self, morpho):
        nbatch = morpho.shape[0]
        # Tpr of affix via binding matrix: affix tpr = F B_affix R^T
        #B_affix = self.morph2affix(morpho).view(nbatch, config.nfill, config.nrole)
        #B_affix = torch.exp(log_softmax(B_affix, dim=1)) # normalize within roles
        #affix = torch.bmm(torch.bmm(F, B_affix), Rt)
        # tpr of affix directly
        affix = self.morph2affix(morpho).view(nbatch, config.dfill, config.drole)
        # Restrict learned affix values to [0,1] or [-1,+1]
        affix = sigmoid(affix) if config.privative_ftrs\
                else tanh(affix)
        #affix.data[0,0] = 1.0 # force affix to begin at 0th position
        #affix = config.seq_embedder.string2tpr('u m', False).unsqueeze(0).expand(nbatch, config.dfill, config.drole)   # xxx testing
        #affix  = tanh(PReLu(affix)) # restrict learned affix components to [0,1]
        #affix = bound_batch(affix)
        #affix = torch.zeros((nbatch, config.dfill, config.drole)) # xxx hack
        unpivot = self.morph2unpivot(morpho)
        unpivot = sigmoid(unpivot).view(nbatch, config.nrole)
        copy = self.morph2copy(morpho)
        copy = sigmoid(copy).view(nbatch, config.nrole)

        if config.discretize:
            affix = torch.round(affix)
            unpivot = torch.round(pivot)
            copy = torch.round(copy)
        
        return affix, unpivot, copy