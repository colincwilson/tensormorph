# -*- coding: utf-8 -*-

import re, sys
import config
from tpr import *
from prosodic_parser import *
import distributions as distrib
#import mcmc


class BiTruncater(nn.Module):
    """
    Combine results of truncating LR-> and <-RL,
    output is a copy vector (nbatch x n)
    maybe: prevent deletion of end delimiter (for use as pivot)
    xxx delegate to pivot finder
    """

    def __init__(self, dcontext=1):
        super(BiTruncater, self).__init__()
        self.dcontext = dcontext
        self.pivots = \
            ['after leftmost C1', 'at leftmost V1', 'after leftmost V1',
             'after leftmost syll', 'after leftmost foot'] + \
            ['before rightmost C1', 'at rightmost V1', 'before rightmost V1',
             'before rightmost syll', 'before rightmost foot']
        self.truncs = ['none'] + [
            dlt + ':' + pvt
            for dlt in ('prefix', 'suffix')
            for pvt in self.pivots
        ]
        self.ntrunc = len(self.truncs)

        self.context2alpha = Linear(dcontext, self.ntrunc)  # deletion logits
        self.context2alpha_begin = Linear(dcontext, 2)  # delete begin delim?
        self.context2alpha_end = Linear(dcontext, 2)  # delete end delim?
        if dcontext == 1:  # remove redundant weight parameters
            self.context2alpha.weight.detach_()
            self.context2alpha_begin.weight.detach_()
            self.context2alpha_end.weight.detach_()
        self.stochastic = []

    def forward(self, form, context):
        nbatch = form.shape[0]
        parser = ProsodicParser()

        # Prosodic parse identifies pivots
        parse_LR = parser(form, 'LR->')
        parse_RL = parser(form, '<-RL')

        # Pivots are located immediately after (LR) or at (RL)
        # prosodic landmarks, except that V1 is used to mark
        # both beginning and end of nucleus (i.e., post-onset, pre-coda)
        pad = torch.zeros(nbatch, 1)
        pivot_LR = {
            x: torch.cat([pad, y[:, :-1]], -1) for x, y in parse_LR.items()
        }
        #pivot_RL = { x : torch.cat([y[:,1:], pad], -1)
        #                for x,y in parse_RL.items() }
        pivot_RL = {'V1': torch.cat([parse_RL['V1'][:, 1:], pad], -1)}

        # Pivots
        pivots = [
            pivot_LR['C1'],  # after leftmost C1
            parse_LR['V1'],  # at leftmost V1
            pivot_LR['V1'],  # after leftmost V1
            pivot_LR['syll1)'],  # after leftmost syll
            pivot_LR['foot1)']
        ]  # after leftmost foot
        pivots += [
            parse_RL['C1'],  # before rightmost C1
            parse_RL['V1'],  # at rightmost V1
            pivot_RL['V1'],  # before rightmost V1
            parse_RL['(syll1'],  # before rightmost syll
            parse_RL['(foot1']
        ]  # before rightmost foot
        pivots = torch.stack(pivots, 1)

        # Delete suffix or prefix relative to each pivot
        delete_suffix = hardtanh0(torch.cumsum(pivots, -1))
        delete_prefix = 1.0 - delete_suffix
        delete = torch.cat(
            [
                torch.zeros(nbatch, 1, config.nrole),  # no deletion
                delete_suffix,  # delete after
                delete_prefix
            ],
            1).detach()  # delete before

        # Invert delete to copy, masking out epsilon fillers
        mask = hardtanh0(form[:, 0, :]).unsqueeze(1)
        copy = (1.0 - delete) * mask

        # Select delete/copy option
        alpha = distrib.rsample(self.context2alpha(context))
        ret = torch.sum(alpha.unsqueeze(-1) * copy, 1)

        # Optionally delete begin or end delim
        alpha_begin = distrib.rsample(self.context2alpha_begin(context))[:, 0]
        alpha_end = distrib.rsample(self.context2alpha_end(context))[:, 0]
        ret = ret * (1.0 - bscalmat(alpha_begin, parse_LR['begin']))
        ret = ret * (1.0 - bscalmat(alpha_end, parse_LR['end']))

        return ret, copy

    def init(self):
        for layer in [
                self.context2alpha, self.context2alpha_begin,
                self.context2alpha_end
        ]:
            layer.weight.data.fill_(0.0)
            layer.bias.data.fill_(0.0)
            layer.bias.data[0] = 1.0
        # Register stochastic params
        self.stochastic = [
            mcmc.StochasticParameter(
                self.context2alpha.bias,
                distrib.SphericalNormal(n=2),  #distrib.Discrete(n=2),
                distrib.Discrete(n=2)),
            mcmc.StochasticParameter(
                self.context2alpha_begin.bias,
                distrib.SphericalNormal(
                    n=2),  #distrib.Discrete(torch.tensor([0.75, 0.25])),
                distrib.Discrete(n=2)),
            mcmc.StochasticParameter(self.context2alpha_end.bias,
                                     distrib.SphericalNormal(n=2), None)
        ]

    def init_(self,
              delete_before=False,
              delete_prefix=False,
              delete_end=False,
              bias=10.0,
              clamp=True):
        for m in [self.delete_prefix, self.delete_begin, self.delete_end]:
            for p in m.parameters():
                p.data.fill_(0.0)
                p.requires_grad = (not clamp)
        self.delete_prefix.bias.data.fill_(
            (1.0 if delete_prefix else -1.0) * bias)
        self.delete_begin.bias.data.fill_(
            (1.0 if delete_begin else -1.0) * bias)
        self.delete_end.bias.data.fill_((1.0 if delete_end else -1.0) * bias)
        self.pivoter.init()

    def show(self, morph):
        """
        Show all truncation options for form
        """
        form = morph.form
        context = torch.zeros(form.shape[0], self.dcontext)
        _, copy = self(form, context)
        #delete = { key : val[0]
        #        for key, val in delete.items()}
        form_segs = morph.form_str[0].split()
        #for key,val in delete.items():
        for i in range(copy.shape[1]):
            #if not re.search('rightmost', self.truncs[i]):
            #    continue
            form_str = ['⟨'+seg+'⟩' if copy[0,i,j]<0.5 else seg \
                for j,seg in enumerate(form_segs)]
            form_str = ' '.join(form_str)
            print(self.truncs[i], form_str)
