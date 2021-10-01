# -*- coding: utf-8 -*-

import config
from tpr import *
from prosodic_parser import *
import distributions as distrib
import mcmc

class BiPivoter(nn.Module):
    """
    Bidirectional pivoter for gradient or stochastic processing
    """
    def __init__(self, dcontext=1):
        super(BiPivoter, self).__init__()
        self.dcontext = dcontext
        self.pivots = ['none', 'after ⋊', 'before ⋉'] \
                      + [' '.join([lcn, drn, ftr])
                            for lcn in ('before', 'after')
                            for drn in ('leftmost', 'rightmost')
                            for ftr in ('C1', 'V1') ]
        self.npivot = len(self.pivots)
        self.context2alpha = Linear(dcontext, self.npivot) # pivot logits
        if dcontext == 1: # remove redundant weight parameters
            self.context2alpha.weight.detach_()
        self.stochastic = []


    def forward(self, form, context):
        nbatch = form.shape[0]
        parser = ProsodicParser()

        parse_LR = parser(form, 'LR->')
        parse_RL = parser(form, '<-RL')

        pad = torch.zeros(nbatch, 1)
        parse_before_LR = { x : torch.cat([y[:,1:], pad], -1)
                for x,y in parse_LR.items() }
        parse_before_RL = { x : torch.cat([y[:,1:], pad], -1)
                for x,y in parse_RL.items() }

        pivots = [
            torch.zeros(nbatch, config.nrole),  # none
            parse_LR['begin'],                  # after ⋊
            parse_before_LR['end'] ]            # before ⋉
        pivots += [lcn[ftr]   # before leftmost/rightmost C1/V1
            for lcn in [parse_before_LR, parse_before_RL]
            for ftr in ['C1', 'V1'] ]
        pivots += [lcn[ftr]   # after leftmost/rightmost C1/V1
            for lcn in [parse_LR, parse_RL]
            for ftr in ['C1', 'V1'] ]
        pivots = torch.stack(pivots, 1).detach()

        alpha = distrib.rsample(
            self.context2alpha(context))
        #print(f'alpha: {alpha.data.numpy()}')
        #print(pivots.shape, alpha.shape)

        pivot = pivots * alpha.unsqueeze(-1)
        pivot = torch.sum(pivot, 1)

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
