#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
from tpr import *

class VocabInserter(nn.Module):
    """
    Vocabulary item look-up implemented with 
    linear map from context (morphosyn + stem scan)
    """
    def __init__(self, dcontext=None):
        super(VocabInserter, self).__init__()
        if dcontext is None:
            dcontext = config.dmorphosyn + 2
        self.context2affix =\
            nn.Linear(dcontext, config.dfill * config.drole, bias=True)
        self.context2unpivot =\
            nn.Linear(dcontext, config.nrole, bias=True)
        self.context2copy =\
            nn.Linear(dcontext, config.nrole, bias=True)
        #self.morph2affix.bias.data.fill_(-2.5)
        #self.morph2unpivot.bias.data.fill_(2.5)

    def forward(self, context):
        nbatch = context.shape[0]
        # Tpr of affix via binding matrix: affix tpr = F B_affix R^T
        #B_affix = self.context2affix(context).view(nbatch, config.nfill, config.nrole)
        #B_affix = torch.exp(log_softmax(B_affix, dim=1)) # normalize within roles
        #affix = torch.bmm(torch.bmm(F, B_affix), Rt)
        # tpr of affix directly
        affix = self.context2affix(context).view(nbatch, config.dfill, config.drole)
        # Restrict learned affix values to [0,1] or [-1,+1]
        affix = sigmoid(affix) if config.privative_ftrs\
                else tanh(affix)
        #affix.data[0,0] = 1.0 # force affix to begin at 0th position
        #affix = config.seq_embedder.string2tpr('u m', False).unsqueeze(0).expand(nbatch, config.dfill, config.drole)   # xxx testing
        #affix  = tanh(PReLu(affix)) # restrict learned affix components to [0,1]
        #affix = bound_batch(affix)
        #affix = torch.zeros((nbatch, config.dfill, config.drole)) # xxx hack
        unpivot = self.context2unpivot(context)
        unpivot = sigmoid(unpivot).view(nbatch, config.nrole)
        copy = self.context2copy(context)
        copy = sigmoid(copy).view(nbatch, config.nrole)

        if config.discretize:
            affix = torch.round(affix)
            unpivot = torch.round(pivot)
            copy = torch.round(copy)
        
        return affix, unpivot, copy