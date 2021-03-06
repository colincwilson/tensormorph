#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
from .tpr import *
from .radial_basis import GaussianPool
from .matcher import Matcher3


class BiScanner(nn.Module):
    """
    Combine results of LR-> and <-RL scanning.
    todo: get local role flag from config
    """
    def __init__(self, morpho_size=1, nfeature=5, npattern=1, node=''):
        super(BiScanner, self).__init__()
        self.morpho_size = morpho_size
        self.nfeature = nfeature
        self.scanner_LR = InhibitoryScanner(morpho_size, nfeature, npattern, direction = 'LR->', node=node+'-LR')
        self.scanner_RL = InhibitoryScanner(morpho_size, nfeature, npattern, direction = '<-RL', node=node+'-RL')
        self.morph2a    = nn.Linear(morpho_size, 1, bias=True)
        self.node = node

    def forward(self, stem, morpho):
        scan_LR = self.scanner_LR(stem, morpho)
        scan_RL = self.scanner_RL(stem, morpho)
        gate_LR = sigmoid(self.morph2a(morpho))
        if config.discretize:
            gate_LR = torch.round(gate_LR)

        scan = gate_LR * scan_LR + (1.0 - gate_LR) * scan_RL

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'gate_LR':gate_LR,
                'scan_LR':scan_LR,
                'scan_RL':scan_RL,
                'scan':scan
            })

        return scan

    def init(self):
        print('BiScanner.init()')
        #self.W0.weight.data.uniform_(-1.0, 0.0)
        #self.W0.weight.data[0,2] = 1.0 # at initial word boundary
        #self.W0.bias.data[:] = -2.0
        #self.W1.weight.data.uniform_(-1.0, 0.0)
        #self.W1.weight.data[0,4] = -1.0 # before final word boundary
        #self.W1.bias.data[:] = -2.0


class InhibitoryScanner(nn.Module):
    """
    Locate leftmost/rightmost/every instance of a pattern with 
    directional inhibition -- assumes that role vectors are local (?)
    """
    def __init__(self, morpho_size, nfeature, npattern=1, direction='LR->', node=''):
        super(InhibitoryScanner, self).__init__()
        self.morpho_size, self.nfeature, self.direction =\
            morpho_size, nfeature, direction
        self.matcher = Matcher3(morpho_size, nfeature, npattern, maxout=(npattern>1), node=node+'-matcher')
        self.morph2c = nn.Linear(morpho_size, 1, bias=True)
        if direction == 'LR->': # inhibit all following positions
            self.W_inhib = -1.0 * torch.ones((config.nrole, config.nrole))
            self.W_inhib = torch.tril(self.W_inhib, diagonal = -1).t()
        if direction == '<-RL': # inhibit all preceding positions
            self.W_inhib = -1.0 * torch.ones((config.nrole, config.nrole))
            self.W_inhib = torch.tril(self.W_inhib, diagonal = -1)
        self.W_inhib = self.W_inhib.detach()
        self.node = node

    def forward(self, stem, morpho):
        c       = torch.exp(self.morph2c(morpho) - 0.0)
        match   = self.matcher(stem, morpho)
        inhib   = torch.matmul(match, self.W_inhib)
        scan    = match * torch.exp(c * inhib)

        if config.discretize:
            scan = torch.round(scan)

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'c':c,
                'match':match,
                'inhib':inhib,
                'scan':scan
            })

        return scan


class RecurrentScanner(nn.Module):
    """
    Locate leftmost/rightmost/every instance of pattern 
    with recurrent gating -- assumes that role vectors are local (?)
    """
    def __init__(self, morpho_size, nfeature, direction='LR->', node=''):
        super(Scanner, self).__init__()
        self.morpho_size, self.nfeature, self.direction =\
            morpho_size, nfeature, direction
        self.matcher = Matcher3(morpho_size, nfeature, npattern=1, maxout=False, node=node+'-matcher')
        self.morph2c = nn.Linear(morpho_size, 1, bias=True)
        if direction == 'LR->':
            self.start, self.end, self.step = 0, config.nrole, 1
        if direction == '<-RL':
            self.start, self.end, self.step = (config.nrole-1), -1, -1
        self.node = node
    
    def forward(self, stem, morpho):
        start, end, step = self.start, self.end, self.step
        nbatch, nrole = stem.shape[0], config.nrole
        c = torch.exp(self.morph2c(morpho) - 0.0).squeeze(-1)
        #c = torch.zeros(nbatch)

        match   = self.matcher(stem, morpho)
        #log_match = torch.log(match) # xxx change return value of matcher xxx watch out for masking!
        scan    = torch.zeros((nbatch, nrole), requires_grad=True).clone()
        h       = torch.zeros(nbatch, requires_grad=True)
        for i in range(start, end, step):
            m = match[:,i]
            s = m * torch.exp(-c*h)
            h = h + m   # alt.: h = h + s -or- h = s + (1.0-s) * h
            scan[:,i] = s

        if config.discretize:
            scan = torch.round(scan)

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'c':c,
                'match':match,
                'scan':scan
            })

        return scan


class BiLSTMScanner(nn.Module):
    """
    Scan stem in both directions with generic RNN cell,
    then concatenate final outputs
    """
    def __init__(self, hidden_size=1):
        super(BiLSTMScanner, self).__init__()
        self.hidden_size = hidden_size
        self.rnn1 = nn.GRUCell(config.dfill, hidden_size, bias=True)
        self.rnn2 = nn.GRUCell(config.dfill, hidden_size, bias=True)
        #self.a = Parameter(torch.zeros(1))

    def forward(self, stem, max_len):
        nbatch, hidden_size = stem.shape[0], self.hidden_size
        rnn1, rnn2 = self.rnn1, self.rnn2
        posn = torch.LongTensor(nbatch).zero_()

        # LR -> scan
        h1_old = torch.zeros((nbatch, hidden_size))
        for i in range(0, max_len, 1):
            f = posn2filler_batch(stem, posn.fill_(i))
            mask = hardtanh(f[:,0], 0.0, 1.0).unsqueeze(1)
            #mask = f[:,0].unsqueeze(1)
            h1_new = rnn1(f, h1_old)
            h1_old = mask * h1_new + (1.0 - mask) * h1_old

        # <-RL scan
        h2_old = torch.zeros((nbatch, hidden_size))
        for i in range(max_len-1, -1, -1):
            f = posn2filler_batch(stem, posn.fill_(i))
            mask = hardtanh(f[:,0], 0.0, 1.0).unsqueeze(1)
            #mask = f[:,0].unsqueeze(1)
            h2_new = rnn2(f, h2_old)
            h2_old = mask * h2_new + (1.0 - mask) * h2_old

        h = torch.cat([h1_old, h2_old], 1)
        return h