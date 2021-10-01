# -*- coding: utf-8 -*-

import config
from tpr import *
from scanner import *
import distributions as distrib
#import mcmc
verbose = 1


class ProsodicPivoter(nn.Module):
    """
    Locate affix insertion sites with prosodic constraints
    (McCarthy & Prince 1993, et seq.)
    """

    def __init__(self):
        super(ProsodicPivoter, self).__init__()
        dcontext = config.dcontext
        self.constraints = [
            'AlignAffixL', 'AlignAffixR', 'AlignStemL', 'AlignStemR', 'Onset',
            'NoCoda'
        ]
        self.nconstraint = len(self.constraints)
        self.context2w = Linear(dcontext, self.nconstraint)  # weights
        self.context2alpha = Linear(dcontext, 1)  # edge bias

    def forward(self, Stem, Affix, context):
        # Prosodic features of stem
        # todo: delegate to syllable_parser
        # todo: move computations to log-linear domain?
        stem = Stem.form
        nbatch = stem.shape[0]
        stem_ftr = unbind_ftr(stem, 0, 3)  # Sym, begin/end, C/V features
        stem_mask = epsilon_mask(stem)  # Epsilon mask
        stem_begin = exp(stem_mask +
                         match_pos(stem_ftr[:, 1]))  # Initial delim (⋊)
        stem_end = exp(stem_mask + match_neg(stem_ftr[:, 1]))  # Final delim (⋉)
        stem_c = exp(stem_mask + match_pos(stem_ftr[:, 2]))  # Consonant (C)
        stem_v = exp(stem_mask + match_neg(stem_ftr[:, 2]))  # Vowel (V)
        stem_begin_prev = shift1(stem_begin,
                                 1)  # Begin delim before current position
        stem_end_next = shift1(stem_end, -1)  # End delim after current position
        stem_v_next = shift1(stem_v, -1)  # V after current position

        # Prosodic features of affix form
        affix = Affix.form
        affix_pivot = Affix.pivot
        affix_copy = Affix.copy
        affix_before_pivot = 1.0 - scan(
            affix_pivot, direction='LR->', inclusive=0)
        affix_ftr = unbind_ftr(affix, 0, 3)  # Sym, begin/end, C/V features
        affix_mask = log(affix_before_pivot * affix_copy)  # Epsilon mask
        affix_c = exp(affix_mask + match_pos(affix_ftr[:, 2]))  # Consonant (C)
        affix_v = exp(affix_mask + match_neg(affix_ftr[:, 2]))  # Vowel (V)
        affix_begin, affix_end = inhibit(exp(affix_mask))
        # Marginals
        affix_c_initial = bdot(affix_begin, affix_c)  # Initial C or V
        affix_v_initial = 1.0 - affix_c_initial  # bdot(affix_begin, affix_v)
        affix_c_final = bdot(affix_end, affix_c)  # Final C or V
        affix_v_final = 1.0 - affix_c_final  # bdot(affix_end, affix_v)

        # Alignment violations
        prefix = 1.0 - stem_begin  # Align(Affix, L, PrWd, L)
        suffix = 1.0 - stem_end_next  # Align(Affix, R, PrWd, R)
        alignL = stem_begin  # Align(Stem, L, PrWd, L)
        alignR = stem_end_next  # Align(Stem, R, PrWd, R)

        # Onset violations (V must be preceded by C)
        onset_in = (1.0 - stem_c) * stem_v_next
        onset_out = (1.0 - stem_c) * affix_v_initial + \
                    (1.0 - affix_c_final) * stem_v_next
        onset = relu(onset_out - onset_in)  # "Do no harm"

        # Coda violations (C must be followed by V;
        # first C of stem-initial onset cluster exempted)
        nocoda_in = (1.0 - stem_begin_prev) * stem_c * (1.0 - stem_v_next)
        nocoda_out = (1.0 - stem_begin_prev) * stem_c * affix_c_initial + \
                     affix_c_final * (1.0 - stem_v_next)
        nocoda = relu(nocoda_out - nocoda_in)  # "Do no harm"

        # Log-linear interaction of prosodic constraints
        # Constraint violations
        violn = torch.stack([prefix, suffix, alignL, alignR, onset, nocoda], -1)
        # Non-negative constraint weights
        w = exp(self.context2w(context))
        # Log-linear combination
        pivot_score = exp(bmatvec(-violn, w))  # exp(-w @ violn)

        # Directional inhibition
        pivot_LR, pivot_RL = inhibit(pivot_score, mask=stem_mask)

        # Convex combination of LR-> and <-RL scans
        alpha = sigmoid(self.context2alpha(context))
        pivot = alpha * pivot_LR + (1.0 - alpha) * pivot_RL

        if verbose:
            stem_str = Stem._str()[0]
            affix_str = Affix._str()[0]
            print(f'stem: {stem_str}')
            print(f'stem_c: {np_round(stem_c[0])}')
            print(f'stem_v: {np_round(stem_v[0])}')
            print(f'affix: {affix_str}')
            print(f'affix_pivot: {np_round(affix_pivot[0])}')
            print(f'affix_before_pivot: {np_round(affix_before_pivot[0])}')
            print(f'affix_mask: {np_round(affix_mask[0])}')
            print(f'affix_c: {np_round(affix_c[0])}')
            print(f'affix_v: {np_round(affix_v[0])}')
            print(f'affix_begin: {np_round(affix_begin[0])}')
            print(f'affix_end: {np_round(affix_end[0])}')
            print(f'affix_c_initial: {np_round(affix_c_initial[0])}')
            print(f'affix_v_initial: {np_round(affix_v_initial[0])}')
            print(f'affix_c_final: {np_round(affix_c_final[0])}')
            print(f'affix_v_final: {np_round(affix_v_final[0])}')
            print(f'w: {np_round(w[0])}')
            print(f'alpha: {np_round(alpha[0])}')
            print(f'pivot_score: {np_round(pivot_score[0])}')
            #sys.exit(0)
        return pivot
