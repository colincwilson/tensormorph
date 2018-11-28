#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
import tpr
from tpr import *
from distance import euclid_batch, euclid_squared_batch


# Map tensor product representation and discrete position to a
# log probability distribution over discrete fillers, or
# iteratively select max prob filler in each discrete position
class Decoder(nn.Module):
    def __init__(self, tau=None):
        super(Decoder, self).__init__()
        self.tau = Parameter(torch.ones(1)*tau) if tau\
              else Parameter(torch.zeros(1))
        #self.tau = Parameter(torch.zeros(1), requires_grad=False)  # xxx clamped precision
        self.eps = torch.zeros(1)+1.0e-8
        self.debug = 0

    def forward(self, T):
        nbatch = T.shape[0]
        F, nfill, dfill, nrole = config.F, config.nfill, config.dfill, config.nrole
        tau = torch.exp(self.tau) + config.tau_min
        eps = self.eps

        posn = torch.LongTensor(nbatch).zero_()
        sim = torch.zeros((nbatch, nfill, nrole), requires_grad=True)
        for k in range(nrole):
            posn.fill_(k)
            f = posn2filler_batch(T, posn)
            #dist = F.expand(nbatch, dfill, nfill) - f.unsqueeze(-1)
            dist = F.unsqueeze(0) - f.unsqueeze(-1)
            dist = (dist**2.0).sum(1)
            sim[:,:,k] = -tau*dist + eps  # xxx disallowed in-place op?
        return sim

    def decode(self, T):
        nbatch = T.shape[0]
        trace  = [ [] for i in range(nbatch)]
        sim = self.forward(T)
        # xxx vectorize over batch?
        for i in range(nbatch):
            pred_argmax = sim[i,:,:].max(0)[1]
            trace.append( pred_argmax.data.numpy() )
        return trace


# Vectorized decoder -- assumes local role vectors
# xxx consolidate with above, using change of role basis
class LocalistDecoder(nn.Module):
    def __init__(self, tau=None):
        super(LocalistDecoder, self).__init__()
        self.tau = Parameter(torch.ones(1)*tau) if tau\
              else Parameter(torch.zeros(1))
        #self.tau = Parameter(torch.zeros(1), requires_grad=False)  # xxx clamped precision
        self.eps = torch.zeros(1)+1.0e-8
        self.debug = 0
        self.decoder = Decoder() if self.debug else None

    def forward(self, T):
        tau = torch.exp(self.tau) + config.tau_min
        eps = self.eps
        dist = euclid_squared_batch(T, config.F)
        sim = -tau*dist + eps
        if self.debug: # verify against generic decoder
            sim_ = self.decoder(T)
            print(sim[0,:,0])
            print(sim_[0,:,0])
            print(np.max(np.abs( sim.data.numpy() - sim_.data.numpy() )))
            sys.exit(0)
        return sim
    
    def decode(self, T):
        nbatch = T.shape[0]
        sim = self.forward(T)
        trace = []
        # xxx vectorize over batch?
        for i in range(nbatch):
            pred_argmax = sim[i,:,:].max(0)[1]
            trace.append( pred_argmax.data.numpy() )
        return trace