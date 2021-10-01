# -*- coding: utf-8 -*-

import config
from tpr import *
import distributions as distrib
from inhibition import *
#import mcmc
verbose = 1

class ProsodicPivoter(nn.Module):
    """
    Locate affix insertion sites with prosodic constraints
    (McCarthy & Prince 1993, et seq.)
    """

    def __init__(self, dcontext=1):
        super(ProsodicPivoter, self).__init__()
        self.constraints = [
            'AlignAffixL', 'AlignAffixR',
            'AlignStemL', 'AlignStemR',
            'Onset', 'NoCoda']
        self.nconstraint = len(self.constraints)
        self.context2w = Linear(dcontext, self.nconstraint) # weights
        self.context2alpha = Linear(dcontext, 1) # edge bias
        self.inhibit = DirectionalInhibition()


    def forward(self, Stem, Affix, context):
        # Prosodic features of stem
        stem = Stem.form
        nbatch = stem.shape[0]
        stem_ftr = unbind_ftr(stem, 0, 3) # Sym, begin/end, C/V features
        stem_mask = hardtanh0(stem_ftr[:,0]) # Epsilon mask
        stem_begin = hardtanh0(stem_mask * stem_ftr[:,1]) # Initial delim
        stem_end = hardtanh0(stem_mask * -stem_ftr[:,1]) # Final delim
        stem_c = hardtanh0(stem_mask * stem_ftr[:,2]) # C
        stem_v = hardtanh0(stem_mask * -stem_ftr[:,2]) # V
        stem_end_next = shift1(stem_end, -1) # End delim after current position
        stem_v_next = shift1(stem_v, -1) # V after current position
        stem_after_v = propagate(stem_v) # Any V before (inclusive) each stem position

        # Prosodic features of affix form
        affix = Affix.form
        affix_pivot = Affix.pivot
        affix_copy = Affix.copy
        affix_before_pivot = 1.0 - propagate(affix_pivot, inclusive=0)
        affix_ftr = unbind_ftr(affix, 0, 3) # Sym, begin/end, C/V features
        affix_mask = hardtanh0(affix_before_pivot * affix_copy) # Segments in affix
        affix_c = hardtanh0(affix_mask * affix_ftr[:,2]) # C
        affix_v = hardtanh0(affix_mask * -affix_ftr[:,2]) # V
        affix_begin, affix_end = self.inhibit(affix_mask)
        affix_c_initial = bdot(affix_begin, affix_c) # Marginals
        affix_v_initial = 1.0 - affix_c_initial # bdot(affix_begin, affix_v)
        affix_c_final = bdot(affix_end, affix_c)
        affix_v_final = 1.0 - affix_c_final # bdot(affix_end, affix_v)
        affix_any_v = propagate(affix_v)[:,-1] # V anywhere in affix

        # Alignment violations
        prefix = 1.0 - stem_begin       # Align(Affix, L, PrWd, L)
        suffix = 1.0 - stem_end_next    # Align(Affix, R, PrWd, R)
        alignL = stem_begin             # Align(Stem, L, PrWd, L)
        alignR = stem_end_next          # Align(Stem, R, PrWd, R)

        # Onset violations
        onset_in = (1.0 - stem_c) * stem_v_next
        onset_out = (1.0 - stem_c) * affix_v_initial + \
                    (1.0 - affix_c_final) * stem_v_next
        onset = relu(onset_out - onset_in)  # "do no harm"

        # Coda violations
        nocoda_in = stem_after_v * stem_c * (1.0 - stem_v_next) # C/_C (after initial complex onset)
        nocoda_out = stem_after_v * stem_c * affix_c_initial + \
                     affix_any_v * affix_c_final * (1.0 - stem_v_next)
        nocoda = relu(nocoda_out - nocoda_in)  # "do no harm"

        # Log-linear weighting of prosodic constraints
        # Assembled constraint violations
        violn = torch.stack([
            prefix, suffix,
            alignL, alignR,
            onset, nocoda], -1)
        # Non-negative constraint weights
        w = torch.exp(self.context2w(context))
        # Log-linear combination
        pivot_score = torch.exp(bmatvec(-violn, w)) # exp(-w @ violn)

        # Directional inhibition
        pivot_LR, pivot_RL = self.inhibit(pivot_score, stem_mask)

        # Convex combination of LR-> and <-RL scans
        alpha = sigmoid(self.context2alpha(context))
        pivot = alpha * pivot_LR + (1.0 - alpha) * pivot_RL

        if verbose:
            stem_str = Stem.print_str()[0]
            affix_str = Affix.print_str()[0]
            print(f'stem: {stem_str}')
            print(f'stem_c: {np_round(stem_c[0])}')
            print(f'stem_v: {np_round(stem_v[0])}')
            print(f'affix: {affix_str}')
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
