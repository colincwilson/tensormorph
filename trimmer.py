#!/usr/bin/env python
# -*- coding: utf-8 -*-
from environ import config
import tpr, radial_basis
from tpr import *
from radial_basis import GaussianPool
from matcher import Matcher3

#import matplotlib
#import matplotlib.pyplot as plt

# combine results of trimming LR-> and <-RL,
# output is a copy vector (nbatch x n)
class BiTrimmer(nn.Module):
    def __init__(self, morpho_size, nfeature=0.0, bias=0.0):
        super(BiTrimmer, self).__init__()
        self.nfeature = nfeature
        self.trimmer_LR = Trimmer(morpho_size, nfeature, direction = 'LR->')
        self.trimmer_RL = Trimmer(morpho_size, nfeature, direction = '<-RL')
        self.a = Parameter(torch.zeros(1))
        #self.morph2a = nn.Linear(config.dmorph, 1, bias=True)

    def forward(self, stem, morpho):
        alpha = sigmoid(self.a)
        copy_LR = self.trimmer_LR(stem, morpho)
        copy_RL = self.trimmer_RL(stem, morpho)
        if config.discretize:
            alpha = torch.round(alpha)

        copy = alpha * copy_LR + (1.0 - alpha) * copy_RL
        return copy

    def init(self):
        print('BiTrimmer.init() does nothing')


# trim a contiguous part of a form by scanning LR-> or <-RL,
# output is a copy vector (nbatch x n) -- assumes local role vectors
class Trimmer(torch.nn.Module):
    def __init__(self, morpho_size=1, nfeature=5, direction='LR->'):
        super(Trimmer, self).__init__()
        self.nfeature    = nfeature
        self.direction   = direction
        self.matcher_phi = Matcher3(morpho_size, nfeature)
        self.matcher_psi = Matcher3(morpho_size, nfeature)
        self.u_phi   = Parameter(torch.zeros(1))
        self.u_psi   = Parameter(torch.zeros(1))
        self.default = Parameter(torch.zeros(1))
        self.copy0   = Parameter(torch.zeros(1))
        if direction == 'LR->':
            self.start, self.end, self.step = 0, config.nrole, 1
        if direction == '<-RL':
            self.start, self.end, self.step = (config.nrole-1), -1, -1

    def forward(self, stem, morpho):
        nbatch, nrole, nfeature = stem.shape[0], config.nrole, self.nfeature
        start, end, step = self.start, self.end, self.step
        u_phi   = torch.exp(self.u_phi).squeeze(-1)
        u_psi   = torch.exp(self.u_psi).squeeze(-1)
        default = torch.sigmoid(self.default)
        copy0   = torch.sigmoid(self.copy0)

        match_phi = self.matcher_phi(stem, morpho)
        match_psi = self.matcher_psi(stem, morpho)

        copy  = torch.zeros((nbatch, nrole), requires_grad=True).clone()
        h_phi = torch.zeros(nbatch, requires_grad=True)
        h_psi = torch.zeros(nbatch, requires_grad=True)
        for k in range(start, end, step):
            p_phi = match_phi[:,k] * (1.0 - h_phi)
            h_phi = p_phi + (1.0 - p_phi) * h_phi
            p_psi = match_psi[:,k] * h_phi * (1.0 - h_psi)
            h_psi = p_psi + (1.0 - p_psi) * h_psi

            theta     = h_phi * (1.0 - h_psi)
            copy[:,k] = theta * (1.0 - default) +\
                        (1.0 - theta) * default

        # keep initial word boundary?
        copy[:,0] = copy0
        copy = copy.clone() # xxx
        # xxx add param for final word boundary?

        if config.discretize:
            copy = torch.round(copy)
        
        return copy
