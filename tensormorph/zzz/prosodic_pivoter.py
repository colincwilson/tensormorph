# -*- coding: utf-8 -*-
# xxx ProsodicMorphologyPivoter

import config
from tpr import *
import distributions as distrib
from inhibition import DirectionalInhibition
#import mcmc
verbose = 0

class ProsodicPivoter(nn.Module):
    """
    Locate affix insertion sites with prosodic constraints
    (see McCarthy & Prince 1993, et seq.)
    """

    def __init__(self, dcontext=1):
        super(ProsodicPivoter, self).__init__()
        self.nprosody = 6 # prefix, suffix, alignL, alignR, onset, nocoda
        self.context2w = Linear(dcontext, self.nprosody) # weights
        self.context2alpha = Linear(dcontext, 1) # directional bias
        self.inhibiter = DirectionalInhibition()
    
    def forward(self, Stem, Affix, context):
        # Prosodic features of stem form
        stem = Stem.form
        nbatch = stem.shape[0]
        stem_ftr = unbind_ftr(stem, 0, 3) # Sym, begin/end, C/V features
        stem_mask = hardtanh0(stem_ftr[:,0]) # Epsilon mask xxx apply hardtanh0?
        stem_begin = hardtanh0(stem_ftr[:,1]) # Initial delim
        stem_end = hardtanh0(-stem_ftr[:,1]) # Final delim
        stem_c = hardtanh0(stem_ftr[:,2]) # C
        stem_v = hardtanh0(-stem_ftr[:,2]) # V
        stem_end_next = shift1(stem_end, -1) # Advance (lead) stem_end by one
        stem_v_next = shift1(stem_v, -1) # Advance (lead) stem_v by one

        # Prosodic features of affix form (convention: left-aligned)
        affix = Affix.form
        affix_pivot = Affix.pivot
        affix_ftr = unbind_ftr(affix, 0, 3) # Sym, begin/end, C/V features
        affix_mask = affix_ftr[:,0] # Epsilon mask xxx apply hardtanh0?
        affix_c = affix_mask * hardtanh0(affix_ftr[:,2])
        affix_v = affix_mask * hardtanh0(-affix_ftr[:,2])
        #affix_mask = affix[:,0,:]
        #affix_c = affix_mask * hardtanh(affix[:,2,:], 0.0, 1.0)
        #affix_v = affix_mask * hardtanh(-affix[:,2,:], 0.0, 1.0)
        affix_c_initial, affix_v_initial, affix_c_final, affix_v_final = \
            self.affix_initial_final(Affix)
        if verbose or 1:
            print(f'affix_c_initial: {affix_c_initial[0]}')
            print(f'affix_v_initial: {affix_v_initial[0]}')
            print(f'affix_c_final: {affix_c_initial[0]}')
            print(f'affix_v_final: {affix_v_initial[0]}')

        # Alignment constraint violations
        prefix = 1.0 - stem_begin       # Align(Affix,L,PrWd,L)
        suffix = 1.0 - stem_end_next    # Align(Affix,R,PrWd,R)
        alignL = stem_begin             # Align(Stem,L,PrWd,L)
        alignR = stem_end_next          # Align(Stem,R,PrWd,R)
        if verbose:
            print(f'prefix: {prefix[0]}')
            print(f'suffix: {suffix[0]}')
            print(f'alignL: {alignL[0]}')
            print(f'alignR: {alignR[0]}')

        # Onset violated by ¬C_stem+V_affix or V_affix+V_stem
        # xxx check
        onset = (1.0 - stem_c) * affix_v_initial + \
                affix_v_final * stem_v_next

        # NoCoda violated by C_stem+C_affix or C_affix+¬V_stem
        nocoda = stem_c * affix_c_initial + \
                 affix_c_final * (1.0 - stem_v_next)
        if verbose or 1:
            print(f'onset: {onset[0]}')
            print(f'nocoda: {nocoda[0]}')

        # Log-linear weighting of prosodic constraints
        #print(prefix.shape, suffix.shape, alignL.shape, alignR.shape)
        #print(onset.shape, nocoda.shape)
        prosodic_pivots = torch.stack([
            prefix, suffix,
            alignL, alignR,
            onset, nocoda], -1)
        w = torch.exp(self.context2w(context)).unsqueeze(1) # non-negative w
        pivot_score = torch.exp(torch.sum(-w * prosodic_pivots, -1)) # exp(-w @ f)
        pivot_score = stem_mask * pivot_score # mask out empty stem positions!
        if verbose:
            print(f'pivot_score: {pivot_score[0]}')

        # Directional inhibition
        pivot_LR, pivot_RL = self.inhibiter(pivot_score, stem_mask)
        if verbose:
            print(f'pivot_LR: {np_round(pivot_LR[0])}')
            print(f'pivot_RL: {np_round(pivot_RL[0])}')

        # Convex combination of LR-> and <-RL scans
        alpha = sigmoid(self.context2alpha(context))
        pivot = alpha * pivot_LR + (1.0 - alpha) * pivot_RL
        if verbose:
            print(f'pivot: {pivot[0]}')
            #sys.exit(0)

        print(f'w: {w[0]}')
        print(f'alpha: {alpha[0]}')
        return pivot


    def affix_initial_final(self, Affix):
        """
        Recurrent computation of whether affix is C- or V- initial / final
        """
        affix = Affix.form
        nbatch = affix.shape[0]
        copy = Affix.copy
        pivot = Affix.pivot
        affix_cv = unbind_ftr(affix, 2)
        C = hardtanh0(affix_cv)
        V = hardtanh0(-affix_cv)
        affix_prosody = torch.stack([copy, pivot, C, V], 1)
        print(f'affix_prosody (copy, pivot, C, V):\n \
            {affix_prosody[0].data[:,0:5]}')

        C_initial = torch.zeros(nbatch, device = config.device)
        V_initial = torch.zeros(nbatch, device = config.device)
        C_final = torch.zeros(nbatch, device = config.device)
        V_final = torch.zeros(nbatch, device = config.device)
        copy_flag = torch.zeros(nbatch, device = config.device)
        pivot_flag = torch.zeros(nbatch, device = config.device)
        for i in range(0,5): # xxx max. affix length!
            Ci, Vi, copyi, pivoti = \
                C[:,i], V[:,i], copy[:,i], pivot[:i]
            C_initial = C_initial + \
                (1.0 - copy_flag) * (1.0 - pivot_flag) * copyi * Ci
            V_initial = V_initial + \
                (1.0 - copy_flag) * (1.0 - pivot_flag) * copyi * Vi
            C_final = \
                hardtanh0((1.0 - copyi) + pivot_flag) * C_final + \
                copyi * (1.0 - pivot_flag) * Ci
            V_final = \
                hardtanh0((1.0 - copy[:,i]) + pivot_flag) * V_final + \
                copyi * (1.0 - pivot_flag) * Vi
            copy_flag = copy_flag + (1.0 - copy_flag) * copy[:,i]
            pivot_flag = pivot_flag + (1.0 - pivot_flag) * pivot[:,i]
        
        C_initial = C_initial.unsqueeze(-1)
        V_initial = V_initial.unsqueeze(-1)
        C_final = C_final.unsqueeze(-1)
        V_final = V_final.unsqueeze(-1)
        return (C_initial, V_initial, C_final, V_final)