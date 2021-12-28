# -*- coding: utf-8 -*-

import config
from tpr import *
# from torch.nn import RNNCell, GRUCell, LSTMCell
from morph import Morph
#from multilinear import Multilinear
#from layers import ResidualSoftmax
from birnn_pivoter import BiRNNPivoter
from position_pivoter import PositionPivoter
from prosodic_pivoter import ProsodicPivoter
from truncater import BiTruncater
from scanner import propagate
import distributions as distrib
import mcmc
import prior


class AffixVocab(nn.Module):
    """
    Affix look-up with optional stem truncation
    todo: relocate truncation? currently disabled
    todo: remove parameters for values beyond max affix length (= 10)
    """

    def __init__(self, daffix=None, dcontext=None, morphophon=False):
        super(AffixVocab, self).__init__()
        self.max_affix_len = max_affix_len = 10

        self.pivot_type = ['birnn', 'position', 'prosodic'][1]
        if self.pivot_type == 'birnn':
            self.pivoter = BiRNNPivoter()
        elif self.pivot_type == 'position':
            self.pivoter = PositionPivoter()
        elif self.pivot_type == 'prosodic':
            self.pivoter = ProsodicPivoter()
        self.truncater = BiTruncater()

        bias = 1  #(config.dcontext > 1)
        dcontext = config.dcontext
        dmorphosyn = config.dmorphosyn
        daffix = 50  # xxx config option

        # Context -> higher-dimensional affix encoding
        # xxx replace with gated highway layer
        self.Icontext2affix = torch.zeros(dcontext, daffix, requires_grad=False)
        self.Icontext2affix.data[:dcontext, :dcontext] = torch.eye(dcontext)
        self.context2affix = nn.Sequential(  # Single- or multi- layer perceptron
            nn.Linear(dcontext, daffix, bias=bias), nn.Sigmoid())
        self.context2affix[0].weight.data.fill_(0.01)
        self.context2affix[0].bias.data.fill_(-3.0)

        # Affix encoding -> affix TPR
        self.affix2form = nn.Sequential(
            nn.Linear(daffix, config.dsym * max_affix_len, bias=bias))

        # Affix encoding -> affix end/pivot logits
        self.affix2end = nn.Sequential(
            nn.Linear(daffix, max_affix_len, bias=bias))

        # Affix encoding -> affix copy logits
        self.affix2copy = nn.Sequential(
            nn.Linear(daffix, max_affix_len, bias=bias))

        # Affix encoding -> stem pivot logits
        self.affix2Wpivot = nn.Sequential(
            nn.Linear(daffix, self.pivoter.npivot, bias=bias))

        # Affix encoding -> zero-affixation logits
        #self.affix2zero = nn.Sequential(
        #    nn.Linear(daffix, 1, bias=bias))
        #self.affix2zero[0].bias.data.fill_(3.0)

        self.trace = None

    def forward(self, Stem, context):
        nbatch = Stem.form.shape[0]
        max_affix_len = self.max_affix_len

        # Combine context dimensions
        context = torch.cat(context, dim=-1)

        # Affix encoding
        affix = self.context2affix(context) + \
                context @ self.Icontext2affix # residual connection

        # Affix form (unbounded real, pre-tanh domain)
        affix_form = self.affix2form(affix) \
                         .view(nbatch, config.dsym, max_affix_len)
        affix_form = torch.cat([
            affix_form,
            torch.zeros(nbatch, config.dsym, config.nrole - max_affix_len)
        ], -1)

        # Weight of unpivot at each position in affix
        affix_end = self.affix2end(affix)
        affix_end = softmax(affix_end, dim=-1)
        affix_end = torch.cat(
            [affix_end,
             torch.ones(nbatch, config.nrole - max_affix_len)], -1)
        affix_pivot = affix_end

        # String well-formedness: soft-enforce epsilons after pivot
        # xxx localist roles only
        pivot_mask = 1.0 - propagate(
            affix_pivot, direction='LR->', inclusive=False)
        mask = pivot_mask
        #affix_form = affix_form * mask.unsqueeze(1)  # Broadcast over ftrs
        affix_form = affix_form * rearrange(mask, 'b r -> b () r')

        # Weight of symbol copy (vs. deletion) at each position in affix
        affix_copy = self.affix2copy(affix)
        affix_copy = sigmoid(affix_copy)
        affix_copy = torch.cat(
            [affix_copy,
             torch.zeros(nbatch, config.nrole - max_affix_len)], -1)

        Affix = Morph(
            form=affix_form,  # local2distrib(affix_form)
            pivot=affix_pivot,
            copy=affix_copy)

        # Stem tpr (real-valued)
        Stem.form = Stem.form

        # Stem.copy = self.truncater(stem.form, context)[0]
        # Weight of copy (vs. deletion) at each position in stem
        Stem.copy = torch.ones(
            nbatch, config.nrole,
            device=config.device)  # xxx enforce full stem copy

        # Weight of pivot at each position in stem
        #if self.pivot_type != 'prosodic':   # xxx experimental
        #    stem_pivot = self.pivoter(Stem, context)
        #else:
        #    stem_pivot = self.pivoter(Stem, Affix, context)
        W = self.affix2Wpivot(affix)
        A = softmax(W, dim=-1)
        pivots = self.pivoter(Stem)
        stem_pivot = einsum('bp,bpr->br', A,
                            pivots)  # pivots.transpose(1,2) @ A
        #stem_pivot = (1.0 - zero) * stem_pivot
        Stem.pivot = stem_pivot

        # Zero-affixation probability
        #p_zero = self.affix2zero(affix).unsqueeze(-1)
        #p_zero = sigmoid(p_zero)

        # Trace (pivot selection)
        if config.recorder is not None:
            self.W = W
            self.A = A
            self.pivots = pivots

        return Stem, Affix, None

    # xxx not used
    def init(self):
        I = torch.eye(config.dcontext)
        self.context2affix_form[0].weight.data.copy_(I)
        self.context2affix_end[0].weight.data.copy_(I)
        self.context2affix_copy[0].weight.data.copy_(I)

    # xxx broken, not used
    def init_old(self):
        for layer in [self.context2affix_form, self.context2affix_end]:
            layer.weight.data.fill_(0.0)
            layer.bias.data.fill_(0.0)
        # All wildcard affix
        affix_wild = torch.zeros(config.dsym, config.nrole)
        for i in range(config.nrole):
            affix_wild[:, i] = config.F[:, -1]
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
            #         torch.ones(config.dsym * config.nrole),
            #         torch.eye(config.dsym * config.nrole)),
            #     None),
            mcmc.StochasticParameter(
                self.context2affix_end.bias,
                #distrib.Geometric(0.5, n),
                distrib.SphericalNormal(n=config.nrole),
                #distrib.Exponential(1, n),
                distrib.Geometric(0.5, n)
                #distrib.OneStep(n)
            )
        ]
