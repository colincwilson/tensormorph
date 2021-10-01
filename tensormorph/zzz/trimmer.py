# -*- coding: utf-8 -*-

from environ import config
from tpr import *
#from radial_basis import GaussianPool
from matcher import Matcher3

class BiTrimmer(nn.Module):
    """
    Combine results of trimming LR-> and <-RL,
    output is a copy vector (nbatch x n).
    """
    def __init__(self, dcontext, nfeature=0, bias=0.0):
        super(BiTrimmer, self).__init__()
        self.nfeature = nfeature
        self.trimmer_LR = Trimmer(
            dcontext,
            nfeature,
            direction = 'LR->')
        self.trimmer_RL = Trimmer(
            dcontext,
            nfeature,
            direction = '<-RL')
        self.a = Parameter(torch.zeros(1))
        #self.context2a = nn.Linear(dcontext, 1, bias=True)
        self.delete_begin = Parameter(torch.zeros(1))
        self.delete_end = Parameter(torch.zeros(1))

    # todo: 'copy' should be renamed 'delete' 
    # -- see inversion in stem_modifier !
    def forward(self, stem, context):
        alpha = sigmoid(self.a)
        copy_LR = self.trimmer_LR(stem, context)
        copy_RL = self.trimmer_RL(stem, context)
        if config.discretize:
            alpha = torch.round(alpha)

        copy = alpha * copy_LR + (1.0 - alpha) * copy_RL

        # Determine whether begin and/or end boundary is deleted
        # (see phon_features.py for conventions on one-hot
        # coding of begin and end boundary symbols)
        delete_begin = torch.sigmoid(self.delete_begin).unsqueeze(-1)
        delete_end = torch.sigmoid(self.delete_end).unsqueeze(-1)
        copy_begin = delete_begin * stem.narrow(1,1,1).squeeze(1)
        copy_end = delete_end * stem.narrow(1,2,1).squeeze(1)
        #print(copy[0])
        copy = torch.max(copy, copy_begin) # xxx change to addition in 
        copy = torch.max(copy, copy_end)   # log-space followed by sigmoid
        #print(copy[0]); sys.exit(0)

        # xxx testing: only allow trimming of begin and/or end boundaries
        copy = torch.max(copy_begin, copy_end)
        #print(copy_begin[0])
        #print(copy_end[0])
        #print(copy[0])
        return copy

    def init(self):
        print('BiTrimmer.init() does nothing')


class Trimmer(torch.nn.Module):
    """
    Trim a contiguous part of a form by scanning LR-> or <-RL,
    output is a copy vector (nbatch x n)
    xxx assumes local role vectors
    """
    def __init__(self, dcontext=1, nfeature=5, direction='LR->'):
        super(Trimmer, self).__init__()
        self.nfeature    = nfeature
        self.direction   = direction
        self.matcher_phi = Matcher3(
            dcontext,
            nfeature)
        self.matcher_psi = Matcher3(
            dcontext,
            nfeature)
        self.u_phi = Parameter(torch.zeros(1))
        self.u_psi = Parameter(torch.zeros(1))
        self.default = Parameter(torch.zeros(1))
        if direction == 'LR->':
            self.start, self.end, self.step = \
                0, config.nrole, 1
        if direction == '<-RL':
            self.start, self.end, self.step = \
                (config.nrole-1), -1, -1

    def forward(self, stem, context):
        nbatch, nrole, nfeature = stem.shape[0], config.nrole, self.nfeature
        start, end, step = self.start, self.end, self.step
        u_phi = torch.exp(self.u_phi).squeeze(-1)
        u_psi = torch.exp(self.u_psi).squeeze(-1)
        default = torch.sigmoid(self.default)

        match_phi = self.matcher_phi(stem, context)
        match_psi = self.matcher_psi(stem, context)

        copy  = torch.zeros((nbatch, nrole), requires_grad=True).clone()
        h_phi = torch.zeros(nbatch, requires_grad=True)
        h_psi = torch.zeros(nbatch, requires_grad=True)
        # todo: eliminate explicit recurrence
        for k in range(start, end, step):
            p_phi = match_phi[:,k] * (1.0 - h_phi)
            h_phi = p_phi + (1.0 - p_phi) * h_phi
            p_psi = match_psi[:,k] * h_phi * (1.0 - h_psi)
            h_psi = p_psi + (1.0 - p_psi) * h_psi

            theta     = h_phi * (1.0 - h_psi)
            copy[:,k] = theta * (1.0 - default) +\
                        (1.0 - theta) * default

        if config.discretize:
            copy = torch.round(copy)
        
        return copy
