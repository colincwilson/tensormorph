# -*- coding: utf-8 -*-

import config
from tpr import *
import distributions as distrib
import mcmc

class BiPivoter(nn.Module):
    """
    BiPivoter for gradient or stochastic processing
    """
    def __init__(self, dcontext=1, nfeature=5):
        super(BiPivoter, self).__init__()
        self.dcontext = dcontext
        self.nfeature = nfeature
        self.pivots = ['none', 'after ⋊', 'before ⋉'] \
            + [' '.join((lcn,drn,ftr)) for lcn in ('before', 'after') 
                for drn in ('leftmost', 'rightmost') for ftr in ('C', 'V')]
        self.npivot = len(self.pivots)
        self.context2alpha = Linear(dcontext, self.npivot) # pivot logits
        if dcontext == 1: # remove redundant weight parameters
            self.context2alpha.weight.detach_()
        self.stochastic = []


    def forward(self, form, context):
        nbatch = form.shape[0]
        nfeature = self.nfeature
        mask = hardtanh(form[:,0,:], 0.0, 1.0).unsqueeze(1)

        # Locate pivots scanning LR-> (deterministic)
        pivot_LR = torch.zeros(nbatch, nfeature, config.nrole)
        found_LR = torch.zeros(nbatch, nfeature)
        for i in range(config.nrole):
            # xxx short-circuit with where() using mask
            match = (form[:,0:nfeature,i] > 0.5) * mask[:,:,i]
            pivot_LR[:,:,i] = match * (1.0 - found_LR)
            found_LR = found_LR + pivot_LR[:,:,i]
        pivot_LR.detach()
        pivot_before_LR = torch.cat((pivot_LR[:,:,1:],
                            torch.zeros(nbatch, nfeature, 1)), -1)
        #print(pivot_LR); print(pivot_before_LR)

        # Locate pivots scanning <-RL
        pivot_RL = torch.zeros(nbatch, nfeature, config.nrole)
        found_RL = torch.zeros(nbatch, nfeature)
        for i in range(config.nrole-1, 0, -1):
            # xxx short-circuit; use where() with mask
            match = (form[:,0:nfeature,i] > 0.5) * mask[:,:,i]
            pivot_RL[:,:,i] = match * (1.0 - found_RL)
            found_RL = found_RL + pivot_RL[:,:,i]
        pivot_RL.detach_()
        pivot_before_RL = torch.cat((pivot_RL[:,:,1:],
                            torch.zeros(nbatch, nfeature, 1)), -1)
        #print(pivot_RL); print(pivot_before_RL)

        pivots = torch.cat([
            torch.zeros(nbatch, 1, config.nrole), # none
            pivot_LR[:,1,:].unsqueeze(1), # after ⋊
            pivot_RL[:,2,:].unsqueeze(1), # before ⋉
            pivot_LR[:,3:,:], # after leftmost C|V
            pivot_RL[:,3:,:], # after rightmost C|V
            pivot_before_LR[:,3:,:], # before leftmost C|V
            pivot_before_RL[:,3:,:], # before rightmost C|V
        ], 1)

        alpha = distrib.rsample(
            self.context2alpha(context))
        #print(f'alpha: {alpha.data.numpy()}')
        #print(pivots.shape, alpha.shape)

        pivot = pivots * alpha.unsqueeze(-1)
        pivot = torch.sum(pivot,1)
        return pivot, (None, None)


    def init(self):
        self.stochastic = []
        self.context2alpha.weight.data.fill_(0.0)
        self.context2alpha.bias.data.fill_(0.0)
        self.context2alpha.bias.data[0] = 1.0
        # Register stochastic params
        self.stochastic = [
            mcmc.StochasticParameter(
                self.context2alpha.bias,
                distrib.SphericalNormal(n=self.npivot),
                distrib.Discrete(n=self.npivot)),
            ]
