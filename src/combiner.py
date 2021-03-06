#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
from .tpr import *
from .radial_basis import GaussianPool
from .writer import Writer

# Encapsulates main hierarchical attention logic of reading / writing.
# attention distributions are computed from gradient (scalar) indices 
# that are updated over processing steps
# Hierarchical structure of attention:
# - morpheme (stem vs. affix)
# - ordinal position within morpheme (0, 1, ..., nrole-1)
class Combiner(nn.Module):
    def __init__(self, node='combiner'):
        super(Combiner, self).__init__()
        self.morph_attender = GaussianPool(2)
        self.posn_attender  = GaussianPool(config.nrole)
        self.writer         = Writer()
        self.node           = node

    def forward(self, stem, affix, copy_stem, copy_affix, pivot, unpivot, max_len):
        nbatch = stem.shape[0]
        morph_attender = self.morph_attender
        posn_attender  = self.posn_attender
        writer = self.writer
        writer.init(nbatch)

        # Initialize soft indices (all zeros)
        a  = torch.zeros(nbatch, 1, requires_grad=True) # morph (0.0=>stem, 1.0=>affix)
        b0 = torch.zeros(nbatch, 1, requires_grad=True) # position within stem
        b1 = torch.zeros(nbatch, 1, requires_grad=True) # position within affix
        c  = torch.zeros(nbatch, 1, requires_grad=True) # position within output

        for i in range(max_len):
            # Map soft indices to attention distributions
            alpha = morph_attender(a)
            beta0 = posn_attender(b0)
            beta1 = posn_attender(b1)
            omega = posn_attender(c)

            # Get copy, pivot, unpivot probabilities at current stem and affix positions
            theta   = alpha[:,0].unsqueeze(1)
            theta0  = dot_batch(pivot, beta0)
            theta1  = dot_batch(unpivot, beta1)
            delta0  = dot_batch(copy_stem, beta0)
            delta1  = dot_batch(copy_affix, beta1)
            #delta1  = dot_batch(affix.narrow(1,0,1).squeeze(1), beta1) # xxx provisional

            # Update tpr of output
            writer(stem, affix, alpha, beta0, beta1, omega, delta0, delta1)

            if config.recorder is not None:
                config.recorder.update_values(self.node, {
                    'morph_indx':a,
                    'stem_indx':b0,
                    'affix_indx':b1,
                    'output_indx':c,
                    'pivot_prob':theta0,
                    'unpivot_prob':theta1
                    })

            # Update stem/affix selection and position within each morph
            # - Switch morph at (un)pivot points, else stay
            a  = a + theta * theta0 - (1.0 - theta) * theta1
            # - Convex combos of advance within each morpheme and stay
            b0 = (1.0 - theta) * b0 + theta * (b0 + 1.0)
            b1 = theta * b1 + (1.0 - theta) * (b1 + 1.0)
            # - Advance within output only if have copied from stem or affix
            c  = c + theta * delta0 + (1.0 - theta) * delta1
            # xxx reset affix position to 0 after unpivot (allowing for multiple affixation)
            #reset_affix = (1.0-theta)*theta2
            #x_affix = reset_affix*0.0 + (1.0-reset_affix)*x_affix

        output = writer.normalize()
        return output
