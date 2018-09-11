#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr, radial_basis
from tpr import *
from radial_basis import GaussianPool
from matcher import Matcher3


# combine results of scanning LR-> and <-RL
class BiScanner(nn.Module):
    def __init__(self, morpho_size=1, nfeature=5, node=''):
        super(BiScanner, self).__init__()
        self.morpho_size = morpho_size
        self.nfeature = nfeature
        # xxx get local role flag from tpr
        self.scanner_LR = Scanner(morpho_size, nfeature, direction = 'LR->', node=node+'-LR')
        self.scanner_RL = Scanner(morpho_size, nfeature, direction = '<-RL', node=node+'-RL')
        self.morph2a    = nn.Linear(morpho_size, 1, bias=True)
        self.node = node

    def forward(self, stem, morpho):
        alpha = sigmoid(self.morph2a(morpho))
        scan_LR = self.scanner_LR(stem, morpho)
        scan_RL = self.scanner_RL(stem, morpho)
        pivot = alpha * scan_LR + (1.0 - alpha) * scan_RL

        if tpr.recorder is not None:
            tpr.recorder.set_values(self.node, {
                'alpha':alpha,
                'scan_LR':scan_LR,
                'scan_RL':scan_RL,
                'pivot':pivot
            })

        return pivot

    def init(self):
        print 'BiScanner.init()'
        #self.W0.weight.data.uniform_(-1.0, 0.0)
        #self.W0.weight.data[0,2] = 1.0 # at initial word boundary
        #self.W0.bias.data[:] = -2.0
        #self.W1.weight.data.uniform_(-1.0, 0.0)
        #self.W1.weight.data[0,4] = -1.0 # before final word boundary
        #self.W1.bias.data[:] = -2.0


# locate leftmost/rightmost/every instance of pattern -- assumes 
# that role vectors are local so that LocalistMatcher is correct
class Scanner(nn.Module):
    def __init__(self, morpho_size, nfeature, direction='LR->', node=''):
        super(Scanner, self).__init__()
        self.morpho_size = morpho_size
        self.nfeature   = nfeature
        self.direction  = direction
        self.matcher    = Matcher3(morpho_size, nfeature, node=node+'-matcher')
        self.morph2u    = nn.Linear(morpho_size, 1, bias=False)
        if direction == 'LR->':
            self.start, self.end, self.step = 0, tpr.nrole, 1
        if direction == '<-RL':
            self.start, self.end, self.step = (tpr.nrole-1), -1, -1
        self.node = node
    
    def forward(self, stem, morpho):
        start, end, step = self.start, self.end, self.step
        nbatch, nrole = stem.shape[0], tpr.nrole
        u = torch.exp(self.morph2u(morpho)).squeeze(-1)

        match = self.matcher(stem, morpho)
        scan = torch.zeros((nbatch, nrole), requires_grad=True).clone()
        h = torch.zeros(nbatch, requires_grad=True)
        for i in xrange(start, end, step):
            p = match[:,i] * torch.exp(-u*h)
            h = p + (1.0-p) * h
            scan[:,i] = p

        if tpr.recorder is not None:
            tpr.recorder.set_values(self.node, {
                'u':u,
                'match':match,
                'scan':scan
            })

        return scan


# scan stem in both directions with RNN cell,
# then concatenate final outputs
class BiLSTMScanner(nn.Module):
    def __init__(self, hidden_size=1):
        super(BiLSTMScanner, self).__init__()
        self.hidden_size = hidden_size
        self.rnn1 = nn.GRUCell(tpr.dfill, hidden_size, bias=True)
        self.rnn2 = nn.GRUCell(tpr.dfill, hidden_size, bias=True)
        #self.a = Parameter(torch.zeros(1))

    def forward(self, stem, max_len):
        nbatch, hidden_size = stem.shape[0], self.hidden_size
        rnn1, rnn2 = self.rnn1, self.rnn2
        posn = torch.LongTensor(nbatch).zero_()

        # LR -> scan
        h1_old = torch.zeros((nbatch, hidden_size))
        for i in xrange(0, max_len, 1):
            f = posn2filler_batch(stem, posn.fill_(i))
            mask = hardtanh(f[:,0], 0.0, 1.0).unsqueeze(1)
            #mask = f[:,0].unsqueeze(1)
            h1_new = rnn1(f, h1_old)
            h1_old = mask * h1_new + (1.0 - mask) * h1_old

        # <-RL scan
        h2_old = torch.zeros((nbatch, hidden_size))
        for i in xrange(max_len-1, -1, -1):
            f = posn2filler_batch(stem, posn.fill_(i))
            mask = hardtanh(f[:,0], 0.0, 1.0).unsqueeze(1)
            #mask = f[:,0].unsqueeze(1)
            h2_new = rnn2(f, h2_old)
            h2_old = mask * h2_new + (1.0 - mask) * h2_old

        h = torch.cat([h1_old, h2_old], 1)
        return h

