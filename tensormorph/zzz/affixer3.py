# -*- coding: utf-8 -*-

import config
from tpr import *
from morph import Morph
from pivoter2 import BiPivoter
from truncater import BiTruncater
import distributions as distrib
import mcmc


class Affixer(nn.Module):
    """
    Affixation with optional stem truncation
    """
    def __init__(self, dcontext=1, morphophon=False):
        super(Affixer, self).__init__()
        self.max_length = 5
        self.context2affix_length = \
            Linear(dcontext, self.max_length)  # context -> length logits
        self.context2affix_form = \
            Linear(dcontext, config.nfill * self.max_length) # context -> form logits
        self.context2affix_copy = \
            Linear(dcontext, self.max_length) # context -> copy logits
        self.pivoter = BiPivoter(
            dcontext = 1, nfeature = 5)
        self.truncater = BiTruncater(
            dcontext = 1, nfeature = 5)
        if dcontext == 1: # remove redundant weight parameters
            self.context2affix_length.weight.detach_()
            self.context2affix_form.weight.detach_()
            #self.context2affix_copy.weight.detach_()
        self.stochastic = []

    def forward(self, stem, context):
        nbatch = stem.form.shape[0]
        affix_length = self.context2affix_length(context)
        affix_form = self.context2affix_form(context) \
            .unsqueeze(-1).view(nbatch, self.max_length, config.nfill)
        affix_copy = self.context2affix_copy(context)

        affix_end = distrib.rsample(affix_length)
        affix_end = torch.cat(
            [affix_end, torch.zeros(nbatch, config.nrole-self.max_length)], 1)
        tril = torch.tril(torch.ones(config.nrole, config.nrole))
        affix_pivot = bmatvec(tril.unsqueeze(0), affix_end)

        affix_form_ = torch.zeros(nbatch, config.dfill, config.nrole)
        affix_copy_ = torch.ones(nbatch, config.nrole)
        for i in range(self.max_length):
            affix_seg = distrib.rsample(affix_form[:,i])
            #print(affix_seg.shape)
            fi = bmatvec(config.F.unsqueeze(0), affix_seg)
            #print(fi.shape)
            ri = config.R[:,i] \
                .unsqueeze(0).expand(nbatch, config.nrole)
            #print(ri.shape)
            affix_form_ = affix_form_ + \
                torch.bmm(fi.unsqueeze(2), ri.unsqueeze(1))
            affix_copy_[:,i] = distrib.rsample_bin(affix_copy[:,i])

        affix = Morph()
        affix.form = affix_form_
        affix.pivot = affix_pivot
        affix.copy = affix_copy_

        stem.copy = self.truncater(stem.form, context)
        #stem.copy = torch.ones(1, config.nrole) # xxx hard-code full stem copy
        stem.pivot = self.pivoter(stem.form, context)[0]
        #print(stem.copy.shape, stem.pivot.shape); sys.exit(0)
        return stem, affix


    def init(self):
        for layer in [self.context2affix_form, 
                      self.context2affix_end]:
            layer.weight.data.fill_(0.0)
            layer.bias.data.fill_(0.0)
        # All wildcard affix
        affix_wild = torch.zeros(config.dfill, config.nrole)
        for i in range(config.nrole):
            affix_wild[:,i] = config.F[:,-1]
        self.context2affix_form.bias.data = \
            affix_wild.view(-1).data
        self.context2affix_form.weight.detach_()
        self.context2affix_form.bias.detach_()
        # Length-2 affix
        self.context2affix_end.bias.data[0] = 1.0
        n = self.context2affix_end.bias.shape[0]
        # Register stochastic params
        self.stochastic = [
            # mcmc.StochasticParameter(
            #     self.context2affix_form.bias,
            #     distrib.MultivariateNormal(
            #         torch.ones(config.dfill * config.nrole),
            #         torch.eye(config.dfill * config.nrole)),
            #     None),
            mcmc.StochasticParameter(
                self.context2affix_end.bias,
                #distrib.Geometric(0.5, n),
                distrib.SphericalNormal(n=config.nrole),
                #distrib.Exponential(1, n),
                distrib.Geometric(0.5, n)
                #distrib.OneStep(n)
            ) ]
