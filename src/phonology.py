#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
from .tpr import *
from .matcher import Matcher3

# bank of phonological rules applied in parallel, 
# each defined by a soft regex3 and a change vector in [-2,+2]
class PhonoRules(nn.Module):
    def __init__(self, morpho_size, nrule=10, node='phonology'):
        super(PhonoRules, self).__init__()
        self.struc_descrip  = Matcher3(morpho_size, 5, nrule, node='SD')
        self.struc_change   = nn.Linear(morpho_size, config.dfill*nrule, bias=True)
        self.rule_gate      = nn.Linear(morpho_size, nrule, bias=True)
        self.nfeature       = config.dfill
        self.nrule          = nrule
        self.node           = node
    
    def forward(self, X, morpho):
        nbatch          = X.shape[0]
        nfeature, nrule = self.nfeature, self.nrule
        struc_descrip   = self.struc_descrip
        struc_change    = self.struc_change(morpho)\
                            .view(nbatch, nfeature, nrule)
        struc_change    = 2.0 * torch.tanh(struc_change)
        rule_gate       = torch.sigmoid(self.rule_gate(morpho))
        #print (struc_change.data.numpy()); sys.exit(0)

        # Find matches to rule struc descriptions
        matches = struc_descrip(X, morpho)
        # Gate rules
        matches = rule_gate.unsqueeze(1) * matches
        # Calculate parallel changes at all matches
        dX      = torch.matmul(matches, struc_change.transpose(2,1))
        dX      = dX.transpose(2,1) # reshape to align with X
        # Apply changes
        Y       = X + dX

        return Y