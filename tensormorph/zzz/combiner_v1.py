#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
from tpr import *
from radial_basis import GaussianPool
from writer import Writer

class Combiner(nn.Module):
    """
    Encapsulates main hierarchical attention logic of reading / writing.
    Attention distributions are computed from gradient (scalar) indices 
    that are updated monotonically over processing.
    Hierarchical structure of attention:
    - level 0: morpheme [stem, affix]
    - level 1: ordinal position within morpheme [0, 1, ..., nrole-1]
    """
    def __init__(self):
        super(Combiner, self).__init__()
        self.morph_attender = GaussianPool(2)
        self.posn_attender  = GaussianPool(config.nrole)
        self.writer = Writer()
        # Extra steps allowed for writing epsilons, beyond 
        # the maximum output length (max_len) of a batch
        self.epsilon_steps = 2


    def forward(self, Stem, Affix, copy_stem, copy_affix, pivot, unpivot, max_len):
        nbatch = Stem.shape[0]
        morph_attender = self.morph_attender
        posn_attender  = self.posn_attender
        writer = self.writer
        writer.init(nbatch)

        # Initialize soft indices (all zeros)
        a  = torch.zeros(nbatch, 1, requires_grad=True) # index within morphs [stem, affix]
        b0 = torch.zeros(nbatch, 1, requires_grad=True) # position within stem [0...]
        b1 = torch.zeros(nbatch, 1, requires_grad=True) # position within affix [0...]
        c  = torch.zeros(nbatch, 1, requires_grad=True) # position within output [0...]

        for i in range(max_len + self.epsilon_steps):
            # Map soft indices to attention distributions
            alpha = morph_attender(a) # distribution over morphs [stem, affix]
            beta0 = posn_attender(b0) # distribution over positions within stem [0...]
            beta1 = posn_attender(b1) # distribution over positions within affix [0...]
            omega = posn_attender(c) # distribution over positions within output [0...]

            # Get copy, pivot, unpivot probabilities at current stem and affix positions
            theta = alpha[:,0].unsqueeze(1) # stem probability
            theta0 = dot_batch(pivot, beta0) # pivot probability
            theta1 = dot_batch(unpivot, beta1) # unpivot probability
            delta0 = dot_batch(copy_stem, beta0) # advance in output if write from stem
            delta1 = dot_batch(copy_affix, beta1) # advance in output if write from affix

            # Update tpr of output
            writer(Stem, Affix, alpha, beta0, beta1, omega, delta0, delta1)

            if config.recorder is not None:
                config.recorder.update_values(self.node, {
                    'morph_indx': a,
                    'stem_indx': b0,
                    'affix_indx': b1,
                    'output_indx': c,
                    'pivot_prob': theta0,
                    'unpivot_prob': theta1
                    })

            if 0: # sharpen scalar indices by projecting back from distribs
                morph_indx = torch.arange(
                    2, requires_grad=False, dtype=torch.float)
                posn_indx = torch.arange(
                    config.nrole, requires_grad=False)
                a = torch.sum(alpha * morph_indx.unsqueeze(0),
                    -1, keepdim=True)
                b0 = torch.sum(beta0 * posn_indx.unsqueeze(0), 
                    -1, keepdim=True)
                b1 = torch.sum(beta1 * posn_indx.unsqueeze(0), 
                    -1, keepdim=True)
                c = torch.sum(omega * posn_indx.unsqueeze(0), 
                    -1, keepdim=True)

            # Update stem/affix selection and position within each morph
            # Switch morph at (un)pivot points, else stay
            a = a + theta * theta0 - (1.0 - theta) * theta1
            #a = hardtanh(a, 0.0, 1.0) # xxx useful?
            # Convex combos of advance within each morpheme and stay
            b0 = (1.0 - theta) * b0 + theta * (b0 + 1.0)
            b1 = theta * b1 + (1.0 - theta) * (b1 + 1.0)
            # Advance within output only if have copied from stem or affix
            c  = c + theta * delta0 + (1.0 - theta) * delta1
            # xxx reset affix position to 0 after unpivot? (allowing for multiple affixation)
            #reset_affix = (1.0-theta)*theta2
            #x_affix = reset_affix*0.0 + (1.0-reset_affix)*x_affix

        Output = writer.normalize() # xxx don't do this?
        return Output
