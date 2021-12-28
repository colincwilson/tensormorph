# -*- coding: utf-8 -*-

import config
from tpr import *
from morph import Morph
from birnn_pivoter import BiRNNPivoter
from position_pivoter import PositionPivoter
from prosodic_pivoter import ProsodicPivoter


class AffixVocab(nn.Module):
    """
    Affix vocabulary, each affix expressed as a sum 
    over daffix basis functions (separately for form, 
    copy, unpivot, and pivot properties)
    """

    def __init__(self, dcontext, daffix):
        super(AffixVocab, self).__init__()
        self.max_affix_len = max_affix_len = 10

        # Pivot module
        self.pivot_type = ['birnn', 'position', 'prosodic'][1]
        if self.pivot_type == 'birnn':
            self.pivoter = BiRNNPivoter()
        elif self.pivot_type == 'position':
            self.pivoter = PositionPivoter()
        elif self.pivot_type == 'prosodic':
            self.pivoter = ProsodicPivoter()

        # Map from context to affix TPR
        self.context2form = nn.Sequential( \
            nn.Linear(dcontext, daffix), \
            nn.Sigmoid(), \
            nn.Linear(daffix, config.dsym * max_affix_len))

        # Map from context to affix copy logits
        self.context2copy = nn.Sequential( \
            nn.Linear(dcontext, daffix), \
            nn.Sigmoid(), \
            nn.Linear(daffix, max_affix_len))

        # Map from context to affix end/pivot logits
        self.context2end = nn.Sequential( \
            nn.Linear(dcontext, daffix), \
            nn.Sigmoid(), \
            nn.Linear(daffix, max_affix_len))

        # Map from context to stem pivot logits
        self.context2pivot = nn.Sequential( \
            nn.Linear(dcontext, daffix), \
            nn.Sigmoid(), \
            nn.Linear(daffix, self.pivoter.npivot))

        self.trace = None

    def forward(self, Stem, context):
        nbatch = Stem.form.shape[0]
        max_affix_len = self.max_affix_len
        #print(nbatch, max_affix_len)

        # Combine context dimensions
        #context = torch.cat(context, dim=-1)

        # Affix form
        affix_form = self.context2form(context)
        affix_form = affix_form.view(nbatch, config.dsym, max_affix_len)
        affix_form = torch.cat([
            affix_form,
            torch.zeros(nbatch, config.dsym, config.nrole - max_affix_len)
        ], -1)

        # Affix copy
        affix_copy = self.context2copy(context)
        affix_copy = sigmoid(affix_copy)
        affix_copy = torch.cat(
            [affix_copy,
             torch.zeros(nbatch, config.nrole - max_affix_len)], -1)

        # Affix end/pivot
        affix_pivot = self.context2end(context)
        affix_pivot = softmax(affix_pivot, dim=-1)
        affix_pivot = torch.cat(
            [affix_pivot,
             torch.ones(nbatch, config.nrole - max_affix_len)], -1)

        # Affix morph
        Affix = Morph(
            form=affix_form,  # local2distrib(affix_form)
            copy=affix_copy,
            pivot=affix_pivot)

        # Stem TPR (real-valued)
        Stem.form = Stem.form
        Stem.copy = torch.ones(
            nbatch, config.nrole,
            device=config.device)  # xxx enforce full stem copy

        # Stem pivot
        W = self.context2pivot(context)
        A = softmax(W, dim=-1)
        pivots = self.pivoter(Stem)
        stem_pivot = einsum('bp,bpr->br', A,
                            pivots)  # pivots.transpose(1,2) @ A
        #stem_pivot = (1.0 - zero) * stem_pivot
        Stem.pivot = stem_pivot

        # Trace (pivot selection)
        if config.recorder is not None:
            self.W = W
            self.A = A
            self.pivots = pivots

        return Stem, Affix, None

    def init(self):
        pass