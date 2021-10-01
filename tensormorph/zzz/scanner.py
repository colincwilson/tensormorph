#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
from tpr import *
from radial_basis import GaussianPool
from matcher import Matcher3

class BiScanner(nn.Module):
    """
    Combine results of LR-> and <-RL scanning.
    todo: get local role flag from config
    """
    def __init__(self, dcontext=1, nfeature=5, npattern=1):
        super(BiScanner, self).__init__()
        self.dcontext = dcontext
        self.nfeature = nfeature
        self.scanner_LR = InhibitoryScanner(
            dcontext,
            nfeature,
            npattern,
            direction = 'LR->')
        self.scanner_RL = InhibitoryScanner(
            dcontext,
            nfeature,
            npattern,
            direction = '<-RL')
        self.context2a = nn.Linear(
            dcontext, 1, bias=True) # xxx should be bias=False

    def forward(self, Stem, Context):
        scan_LR = self.scanner_LR(Stem, Context)
        scan_RL = self.scanner_RL(Stem, Context)
        gate_LR = sigmoid(self.context2a(Context))
        if config.discretize:
            gate_LR = torch.round(gate_LR)

        scan = gate_LR * scan_LR + (1.0 - gate_LR) * scan_RL

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'gate_LR': gate_LR,
                'scan_LR': scan_LR,
                'scan_RL': scan_RL,
                'scan': scan
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
    directional inhibition
    xxx assumes localist roles (?)
    xxx see now inhibition.DirectionalInhibition
    """
    def __init__(self, dcontext, nfeature, npattern=1, direction='LR->'):
        super(InhibitoryScanner, self).__init__()
        self.dcontext, self.nfeature, self.direction = \
            dcontext, nfeature, direction
        self.matcher = Matcher3(
            dcontext,
            nfeature,
            npattern,
            maxout=(npattern>1))
        self.context2tau = nn.Linear(
            dcontext,
            1,
            bias=True) # xxx should be bias=False
        # todo: clean up inhibition patterns
        if direction == 'LR->': # inhibit all following positions
            self.W_inhib = -1.0 * torch.ones((config.nrole, config.nrole))
            self.W_inhib = torch.tril(self.W_inhib, diagonal = -1).t()
        if direction == '<-RL': # inhibit all preceding positions
            self.W_inhib = -1.0 * torch.ones((config.nrole, config.nrole))
            self.W_inhib = torch.tril(self.W_inhib, diagonal = -1)
        self.W_inhib = self.W_inhib.detach()

    def forward(self, Stem, Context):
        #print(__file__, ' context.shape =', context.shape)
        tau = torch.exp(self.context2tau(context) - 0.0)
        match = self.matcher(Stem, Context)
        inhib = match @ self.W_inhib # torch.matmul(match, self.W_inhib)
        scan = match * torch.exp(tau * inhib) # xxx double exponential!

        if config.discretize:
            scan = torch.round(scan)

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'tau': tau,
                'match': match,
                'inhib': inhib,
                'scan': scan
            })

        return scan


class RecurrentScanner(nn.Module):
    """
    Locate leftmost/rightmost/every instance of pattern 
    with recurrent gating
    xxx assumes that role vectors are local (?)
    """
    def __init__(self, dcontext, nfeature, direction='LR->', node=''):
        super(Scanner, self).__init__()
        self.dcontext, self.nfeature, self.direction =\
            dcontext, nfeature, direction
        self.matcher = Matcher3(
            dcontext,
            nfeature,
            npattern=1,
            maxout=False)
        self.context2tau = nn.Linear(
            dcontext,
            1,
            bias=True)
        if direction == 'LR->':
            self.start, self.end, self.step = 0, config.nrole, 1
        if direction == '<-RL':
            self.start, self.end, self.step = (config.nrole-1), -1, -1
        self.node = node
    
    def forward(self, Stem, Context):
        start, end, step = self.start, self.end, self.step
        nbatch, nrole = Stem.shape[0], config.nrole
        tau = torch.exp(self.context2tau(context) - 0.0).squeeze(-1)
        #tau = torch.zeros(nbatch)

        match = self.matcher(Stem, Context)
        #log_match = torch.log(match) # xxx change return value of matcher xxx watch out for masking!
        scan = torch.zeros((nbatch, nrole), requires_grad=True).clone()
        h = torch.zeros(nbatch, requires_grad=True)
        for i in range(start, end, step):
            m = match[:,i]
            s = m * torch.exp(-c*h)
            h = h + m   # alt.: h = h + s -or- h = s + (1.0-s) * h
            scan[:,i] = s

        if config.discretize:
            scan = torch.round(scan)

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'tau': tau,
                'match': match,
                'scan': scan
            })

        return scan


# xxx use log masking; deprecate or make parallel over roles
class BiLSTMScanner(nn.Module):
    """
    Scan stem in both directions with generic RNN cell,
    then concatenate final outputs
    """
    def __init__(self, hidden_size=1):
        super(BiLSTMScanner, self).__init__()
        self.hidden_size = hidden_size
        self.rnn1 = nn.GRUCell(
            config.dfill,
            hidden_size,
            bias=True)
        self.rnn2 = nn.GRUCell(
            config.dfill,
            hidden_size,
            bias=True)
        #self.a = Parameter(torch.randn(1))

    def forward(self, Stem, max_len):
        nbatch, hidden_size = Stem.shape[0], self.hidden_size
        rnn1, rnn2 = self.rnn1, self.rnn2
        posn = torch.LongTensor(nbatch).zero_()

        # LR -> scan
        h1_old = torch.zeros((nbatch, hidden_size))
        for i in range(0, max_len, 1):
            f = posn2filler_batch(Stem, posn.fill_(i))
            mask = hardtanh(f[:,0], 0.0, 1.0).unsqueeze(1)
            #mask = f[:,0].unsqueeze(1)
            h1_new = rnn1(f, h1_old)
            h1_old = mask * h1_new + (1.0 - mask) * h1_old

        # <-RL scan
        h2_old = torch.zeros((nbatch, hidden_size))
        for i in range(max_len-1, -1, -1):
            f = posn2filler_batch(Stem, posn.fill_(i))
            mask = hardtanh(f[:,0], 0.0, 1.0).unsqueeze(1)
            #mask = f[:,0].unsqueeze(1)
            h2_new = rnn2(f, h2_old)
            h2_old = mask * h2_new + (1.0 - mask) * h2_old

        h = torch.cat([h1_old, h2_old], 1)
        return h