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
            'AlignAffixL', 'AlignAffixR', 'AlignbaseL', 'AlignbaseR', 'Onset',
            'NoCoda'
        ]
        self.nconstraint = len(self.constraints)
        self.context2w = Linear(dcontext, self.nconstraint)  # weights
        self.context2alpha = Linear(dcontext, 1)  # edge bias

    def forward(self, base, affix, context):
        # Prosodic features of base
        # todo: delegate to syllable_parser
        # todo: move computations to log-linear domain?
        base_form = base.form
        nbatch = base_form.shape[0]
        base_ftr = unbind_ftr(base_form, 0, 3)  # Sym, begin/end, C/V features
        base_mask = epsilon_mask(base_form)  # Epsilon mask
        base_begin = exp(base_mask +
                         match_pos(base_ftr[:, 1]))  # Initial delim (⋊)
        base_end = exp(base_mask + match_neg(base_ftr[:, 1]))  # Final delim (⋉)
        base_c = exp(base_mask + match_pos(base_ftr[:, 2]))  # Consonant (C)
        base_v = exp(base_mask + match_neg(base_ftr[:, 2]))  # Vowel (V)
        base_begin_prev = shift1(base_begin,
                                 1)  # Begin delim before current position
        base_end_next = shift1(base_end, -1)  # End delim after current position
        base_v_next = shift1(base_v, -1)  # V after current position

        # Prosodic features of affix
        affix_form = affix.form
        affix_pivot = affix.pivot
        affix_copy = affix.copy
        affix_before_pivot = 1.0 - scan(
            affix_pivot, direction='LR->', inclusive=0)
        affix_ftr = unbind_ftr(affix_form, 0, 3)  # Sym, begin/end, C/V features
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
        prefix = 1.0 - base_begin  # Align(Affix, L, PrWd, L)
        suffix = 1.0 - base_end_next  # Align(Affix, R, PrWd, R)
        alignL = base_begin  # Align(base, L, PrWd, L)
        alignR = base_end_next  # Align(base, R, PrWd, R)

        # Onset violations (V must be preceded by C)
        onset_in = (1.0 - base_c) * base_v_next
        onset_out = (1.0 - base_c) * affix_v_initial + \
                    (1.0 - affix_c_final) * base_v_next
        onset = relu(onset_out - onset_in)  # "Do no harm"

        # Coda violations (C must be followed by V;
        # first C of base-initial onset cluster exempted)
        nocoda_in = (1.0 - base_begin_prev) * base_c * (1.0 - base_v_next)
        nocoda_out = (1.0 - base_begin_prev) * base_c * affix_c_initial + \
                     affix_c_final * (1.0 - base_v_next)
        nocoda = relu(nocoda_out - nocoda_in)  # "Do no harm"

        # Log-linear interaction of prosodic constraints
        # Constraint violations
        violn = torch.stack([prefix, suffix, alignL, alignR, onset, nocoda], -1)
        # Non-negative constraint weights
        w = exp(self.context2w(context))
        # Log-linear combination
        pivot_score = exp(bmatvec(-violn, w))  # exp(-w @ violn)

        # Directional inhibition
        pivot_LR, pivot_RL = inhibit(pivot_score, mask=base_mask)

        # Convex combination of LR-> and <-RL scans
        alpha = sigmoid(self.context2alpha(context))
        pivot = alpha * pivot_LR + (1.0 - alpha) * pivot_RL

        if verbose:
            base_str = base._str()[0]
            affix_str = affix._str()[0]
            print(f'base: {base_str}')
            print(f'base_c: {np_round(base_c[0])}')
            print(f'base_v: {np_round(base_v[0])}')
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
