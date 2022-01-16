# -*- coding: utf-8 -*-

import config
from tpr import *
from morph import Morph
from matcher import Matcher3
from torch.linalg import norm as l2norm
import data_util


class Phonology(nn.Module):
    """
    Weighted phonological rules/constraints for feature change, epenthesis, and deletion
    todo: conditioning on output context, with LR-> or <-RL directional application
    """

    def __init__(self, dcontext=1, npattern=10):
        super(Phonology, self).__init__()
        self.nfeature = nfeature = config.dsym
        self.npattern = npattern

        # Structural description for each pattern
        self.matcher = Matcher3(dcontext, nfeature, npattern)

        # Modification type for each pattern (log-linear)
        self.mod_types = ['chng', 'deln', \
            'epen1_before', 'epen2_before', \
            'epen1_after', 'epen2_after']
        self.nmod = nmod = len(self.mod_types)
        self.mod = nn.Linear(dcontext, npattern * nmod)

        # Modification weight for each pattern (log-linear)
        self.w_chng = nn.Linear(dcontext, npattern * nfeature)
        self.w_deln = nn.Linear(dcontext, npattern)
        self.w_epen = nn.Linear(dcontext, npattern * 4)

        # Modification features for each pattern (pre-tanh)
        self.v_chng = nn.Linear(dcontext, npattern * nfeature)
        self.v_epen = nn.Linear(dcontext, npattern * nfeature * 4)

        # Faithfulness weights (log scale except for w_faith)
        self.w_faith = 0.0  # base weight
        self.w_nochng = nn.Linear(dcontext, nfeature)
        self.w_nodeln = nn.Linear(dcontext, 1)
        self.w_noepen = nn.Linear(dcontext, 1)

        #self.pretrain()

    def forward(self, form, context, input_posns=None):
        """
        Apply bank of phonological patterns to forms (in unbounded, pre-tanh domain) given contexts
        """
        nbatch = form.shape[0]
        nfeature = self.nfeature
        npattern = self.npattern
        nmod = self.nmod
        self.trace = {}

        # Each pattern regulates one modification type
        mod = self.mod(context) \
                .view(nbatch, npattern, nmod)
        self.trace['mod'] = mod.clone().detach()
        mod = torch.softmax(mod, dim=-1)  # [nbatch x npattern x nmod]

        # Modification weights for each pattern
        # (zero => no preference)
        w_chng = self.w_chng(context) \
                    .view(nbatch, npattern, nfeature)
        w_deln = self.w_deln(context) \
                    .view(nbatch, npattern, 1)
        w_epen = self.w_epen(context) \
                    .view(nbatch, npattern, 4)
        self.trace['w_chng'] = w_chng.clone().detach()
        self.trace['w_deln'] = w_deln.clone().detach()
        self.trace['w_epen'] = w_epen.clone().detach()

        w_chng = mod[...,
                     0].unsqueeze(-1) * w_chng  # [nbatch x npattern x nftr]
        w_deln = mod[..., 1].unsqueeze(-1) * w_deln  # [nbatch x npattern x 1]
        w_epen = mod[..., 2:] * w_epen  # [nbatch x npattern x 4]

        # Modification target values for each pattern
        # (all zero => no features changed, epenthesis of epsilon)
        v_chng = self.v_chng(context) \
                    .view(nbatch, npattern, nfeature)
        v_epen = self.v_epen(context) \
                    .view(nbatch, npattern, nfeature, 4)
        self.trace['v_chng'] = v_chng.clone().detach()
        self.trace['v_epen'] = v_epen.clone().detach()

        # Faithfulness weights
        # (constrained to be positive)
        w_faith = self.w_faith
        w_nochng = exp(self.w_nochng(context)) + w_faith  # [nbatch x nfeature]
        w_nodeln = exp(self.w_nodeln(context)) + w_faith  # [nbatch x 1]
        w_noepen = exp(self.w_noepen(context)) + w_faith  # [nbatch x 1]

        # Match patterns at all positions
        # xxx input-based rules only
        match = self.matcher(form, context)  # [nbatch x npattern x nrole]
        self.trace['match'] = match.clone().detach()
        match = exp(match)  # Map real-valued matches into (0,1)

        # Construct output tpr incrementally LR->
        output_posn = torch.zeros((nbatch, 1), device=config.device)
        output = torch.zeros((nbatch, config.dsym, config.nrole),
                             device=config.device)

        # Input positions at which to apply phonology
        if input_posns is None:
            input_posns = range(config.nrole)
        for i in input_posns:  # xxx extra steps?
            # Symbol and pattern match at ith *input* position
            sym_i = form[..., i]  # [nbatch x nftr]
            match_i = match[..., i]  # [nbatch x npattern]

            # Epenthesis before
            w_epen1_before, w_epen2_before = \
                w_epen[...,:2].unbind(-1)
            v_epen1_before, v_epen2_before = \
                v_epen[...,:2].unbind(-1)
            output, output_posn = self.apply_epenthesis(
                match_i, w_epen1_before, w_epen2_before, v_epen1_before,
                v_epen2_before, w_noepen, output, output_posn)

            # Feature-change / deletion / copy
            output, output_posn = self.apply_change(match_i, sym_i, w_chng,
                                                    v_chng, w_deln, w_nochng,
                                                    w_nodeln, output,
                                                    output_posn)

            # Epenthesis after
            w_epen1_after, w_epen2_after = \
                w_epen[...,2:].unbind(-1)
            v_epen1_after, v_epen2_after = \
                v_epen[...,2:].unbind(-1)
            output, output_posn = self.apply_epenthesis(match_i, w_epen1_after,
                                                        w_epen2_after,
                                                        v_epen1_after,
                                                        v_epen2_after, w_noepen,
                                                        output, output_posn)

        return output

    def apply_epenthesis(self, match_i, w_epen1, w_epen2, v_epen1, v_epen2,
                         w_noepen, output, output_posn):
        #print(f'match_i {match_i.shape}, w_epen1 {w_epen1.shape}, w_epen2 {w_epen2.shape}, v_epen1 {v_epen1.shape}, v_epen2 {v_epen2.shape}, w_noepen {w_noepen.shape}')
        # Probability and features of epenthesis-1
        w_epen1 = match_i * w_epen1  # [nbatch x npattern]
        p_epen1 = sigmoid(  # [nbatch x 1]
            torch.sum(w_epen1, dim=1, keepdim=True) - w_noepen)
        w_epen1 = w_epen1.unsqueeze(-1)  # broadcast over features
        v_epen1 = torch.sum(
            relu(w_epen1) * v_epen1,
            dim=1)  # [nbatch x nfeature] xxx document relu

        # Apply epenthesis-1
        # xxx reuse code in morph.py
        posn_attn = config.posn_attender(output_posn)
        output_ = attn_bind(output, v_epen1, posn_attn)[0]
        output = p_epen1.unsqueeze(-1) * output_ + \
                 (1.0 - p_epen1.unsqueeze(-1)) * output
        output_posn = output_posn + p_epen1

        # Probability and features of epenthesis-2
        if 0:  # xxx 0 -> limit epenthesis to one segment at each locus
            w_epen2 = match_i * w_epen2  # [nbatch x npattern]
            p_epen2 = sigmoid(  # [nbatch x 1]
                torch.sum(w_epen2, dim=1, keepdim=True) - w_noepen)
            p_epen2 = p_epen1 * p_epen2  # epenthesis-2 => epenthesis-1
            w_epen2 = w_epen2.unsqueeze(-1)  # broadcast over features
            v_epen2 = torch.sum(
                relu(w_epen2) * v_epen2,
                dim=1)  # [nbatch x nfeature] xxx document relu

            # Apply epenthesis-2
            posn_attn = config.posn_attender(output_posn)
            output_ = attn_bind(output, v_epen2, posn_attn)[0]
            output = p_epen2.unsqueeze(1) * output_ + \
                    (1.0 - p_epen2.unsqueeze(1)) * output
            output_posn = output_posn + p_epen2

        #print(output.shape, output_posn.shape)
        return output, output_posn

    def apply_change(self, match_i, sym_i, w_chng, v_chng, w_deln, w_nochng,
                     w_nodeln, output, output_posn):
        #print(match_i.shape, sym_i.shape, w_chng.shape, v_chng.shape, w_deln.shape, no_chng.shape, no_deln.shape); sys.exit(0)
        # Change probability and change for each feature
        w_chng = match_i.unsqueeze(-1) * w_chng
        p_chng = sigmoid(torch.sum(w_chng, dim=1) - w_nochng)
        v_chng = torch.sum(relu(w_chng) * v_chng, dim=1)  # xxx document relu

        # Apply change / copy
        # xxx torch.abs(v_chng) is incorrect because
        # v_chng could be zero, instead want to identify
        # elements of sym_i that are distinct from target v_chng
        #sym = (1.0 - p_chng) * sym_i + p_chng * (sym_i + (v_chng - sym_i))
        sym = sym_i + p_chng * (v_chng - sym_i)
        posn_attn = config.posn_attender(output_posn)
        output_ = attn_bind(output, sym, posn_attn)[0]

        # Deletion probability (affects all features)
        w_deln = match_i.unsqueeze(-1) * w_deln  # [nbatch x npattern x 1]
        p_deln = sigmoid(torch.sum(w_deln, dim=1) - w_nodeln)  # [nbatch x 1]

        # Apply deletion
        output = (1.0 - p_deln.unsqueeze(-1)) * output_ + \
                 p_deln.unsqueeze(-1) * output
        output_posn = output_posn + (1.0 - p_deln)  # advance iff do not delete

        #print(output.shape, output_posn.shape)
        return output, output_posn

    def pretrain(self, theta=0.99):
        """
        Tune w_faith to enforce default quasi-identity map
        """
        print('Tuning phonology.w_faith for identity map')
        tau_min_orig = config.posn_attender.tau_min
        config.posn_attender.tau_min = 1.5  # xxx
        print(f'using posn_attender.tau_min = {config.posn_attender.tau_min}')

        batch = data_util.morph_batcher(config.dat_train)
        inpt = batch['stem'].form
        cntxt = torch.zeros((inpt.shape[0], 1), config.device)

        print('\tw_faith\t\tprop. identity')
        w_faith_tune = None
        for w_faith in np.arange(0.0, 8.0, .25):
            self.w_faith = w_faith
            outpt = self(inpt, cntxt)
            outpt = Morph(outpt)._str()
            prop_ident = np.mean([
                int(outpt[i] == batch['stem_str'][i])
                for i in range(len(outpt))
            ])
            print(f'\t{w_faith:0.2f}\t\t{prop_ident}')
            print('\texample:', batch['stem_str'][0], '->', outpt[0])
            if prop_ident > theta:
                break
            w_faith_tune = w_faith
        config.posn_attender.tau_min = tau_min_orig


# # # # # Deprecated # # # # #


class Phonology1(nn.Module):
    """
    Bank of weighted phonological rules / constraints
    """

    def __init__(self, dcontext=1, npattern=1):
        super(Phonology1, self).__init__()
        self.npattern = npattern
        self.nfeature = nfeature = config.dsym
        # Structural Description (SD)
        self.matcher = Matcher3(dcontext, nfeature, npattern)
        #self.matchers = {
        #    'matcher_prev': LiteralMatcher(dcontext, nfeature, nrule),
        #    'matcher_cntr': LiteralMatcher(dcontext, nfeature, nrule),
        #    'matcher_next': LiteralMatcher(dcontext, nfeature, nrule) }
        # Structural Change (SC)
        self.change_types = {'mod', 'deln', 'epen1', 'epen2'}
        self.nchange = nchange = len(self.change_types)
        self.change = nn.Linear(dcontext, nchange * npattern)
        self.ftr1 = nn.Linear(dcontext, nfeature * npattern)
        self.ftr2 = nn.Linear(dcontext, nfeature * npattern)
        # Pattern weight
        self.W = nn.Linear(dcontext, npattern)

    def forward(self, form, context):
        nbatch = form.shape[0]
        npattern = self.npattern
        nfeature = self.nfeature
        nchange = self.nchange

        match = self.matcher(form, context)  # [nbatch x npattern x nrole]
        change = self.change(context).view(nbatch, nchange, npattern)
        change = softmax(change, dim=1)  # [nbatch x nchange x npattern]
        ftr1 = self.ftr1(context).view(nbatch, nfeature, npattern)
        ftr1 = torch.tanh(ftr1)
        ftr2 = self.ftr2(context).view(nbatch, nfeature, npattern)
        ftr2 = torch.tanh(ftr2)
        w = exp(self.W(context).view(nbatch, npattern))

        output_posn = torch.zeros((nbatch, 1), device=config.device)
        output = torch.zeros((nbatch, config.dsym, config.nrole),
                             device=config.device)

        for i in range(config.nrole):  # xxx extra steps
            # Pattern probabilities at input position i
            match_i = match[..., i]  # [nbatch x npattern]
            Z = torch.sum(w * match_i, 1, keepdim=True) + 1.0
            apply_i = (w * match_i) / Z  # xxx unstable softmax?
            apply_i = apply_i.unsqueeze(1)  # [nbatch x 1 x npattern]
            p_nochange_i = 1.0 / Z  # [nbatch x 1] # xxx incorrect?
            p_nochange_i = p_nochange_i.unsqueeze(1)  # [nbatch x 1 x 1]

            # Weighted sum of changes across matched patterns
            ftr1_i = torch.sum(apply_i * ftr1, -1)  # [nbatch x nfeature]
            ftr2_i = torch.sum(apply_i * ftr2, -1)  # [nbatch x nfeature]
            change_i = torch.sum(apply_i * change, -1)  # [nbatch x nchange]

            # Change probabilities
            change_i = change_i.unsqueeze(-1).unsqueeze(-1)
            # [nbatch x nchange x 1 x 1]
            p_mod_i = change_i[:, 0]  # [nbatch x 1 x 1]
            #p_deln_i = change_i[:,1] # (implicit)
            p_epen1_i = change_i[:, 2]
            p_epen2_i = change_i[:, 3]

            # Copy / mod / epen feature vectors
            copy_i = form[..., i]  # [nbatch x nfeature]
            mod_i = copy_i * (1.0 - torch.abs(ftr1_i)) + ftr1_i  # apply mod
            epen1_i = ftr1_i  # [nbatch x nfeature]
            epen2_i = ftr2_i  # [nbatch x nfeature]

            # Apply weighted superposition of changes
            # Copy
            posn_attn = config.posn_attender(output_posn)
            output_ = attn_bind(output, copy_i, posn_attn)[0]
            p_copy_i = (p_nochange_i + p_epen1_i + p_epen2_i)
            output = p_copy_i * output_ + (1.0 - p_copy_i) * output
            output_posn = output_posn + p_copy_i.squeeze(-1)

            # Epenthesis1 (applies after match)
            posn_attn = config.posn_attender(output_posn)
            output_ = attn_bind(output, epen1_i, posn_attn)[0]
            output_posn_ = output_posn + 1.0
            output = p_epen1_i * output_ + (1.0 - p_epen1_i) * output
            output_posn = p_epen1_i.squeeze(-1) * output_posn_ + \
                            (1.0 - p_epen1_i.squeeze(-1)) * output_posn

            # Epenthesis2 (applies after match)
            posn_attn = config.posn_attender(output_posn)
            output_ = attn_bind(output, epen1_i, posn_attn)[0]
            output_posn_ = output_posn + 1.0

            posn_attn = config.posn_attender(output_posn_)
            output__ = attn_bind(output_, epen2_i, posn_attn)[0]
            output_posn__ = output_posn_ + 1.0

            output = p_epen2_i * output__ + (1.0 - p_epen2_i) * output
            output_posn = p_epen2_i.squeeze(-1) * output_posn__ + \
                            (1.0 - p_epen2_i.squeeze(-1)) * output_posn

            # Mod
            posn_attn = config.posn_attender(output_posn)
            output_ = attn_bind(output, mod_i, posn_attn)[0]
            output = p_mod_i * output_ + (1.0 - p_mod_i) * output
            output_posn = output_posn + p_mod_i.squeeze(-1)

            # Deln
            # (write nothing and do not advance output posn)

        #output = Morph(output)
        return output


class PhonoRules(nn.Module):
    """
    Bank of phonological rules applied in parallel, each 
    defined by a soft regex3 and a change vector in [-2,+2]
    """

    def __init__(self, morpho_size, nrule=10):
        super(PhonoRules, self).__init__()
        self.struc_descrip = Matcher3(morpho_size, 5, nrule, node='SD')
        self.struc_change = nn.Linear(
            morpho_size, config.dsym * nrule, bias=True)
        self.rule_gate = nn.Linear(morpho_size, nrule, bias=True)
        self.nfeature = config.dsym
        self.nrule = nrule

    def forward(self, X, morpho):
        nbatch = X.shape[0]
        nfeature, nrule = self.nfeature, self.nrule
        struc_descrip = self.struc_descrip
        struc_change    = self.struc_change(morpho)\
                            .view(nbatch, nfeature, nrule)
        struc_change = 2.0 * tanh(struc_change)
        rule_gate = sigmoid(self.rule_gate(morpho))
        #print(struc_change.data.numpy()); sys.exit(0)

        # Find matches to rule struc descriptions
        matches = struc_descrip(X, morpho)
        # Gate rules
        matches = rule_gate.unsqueeze(1) * matches
        # Calculate parallel changes at all matches
        dX = matches @ struc_change.transpose(
            2, 1)  #torch.matmul(matches, struc_change.transpose(2,1))
        dX = dX.transpose(2, 1)  # reshape to align with X
        # Apply changes
        Y = X + dX

        return Y


# Encode a single (currently hand-specified) rule
# for demo/testing purposes
# Format for rule application:
#   output = input + locus_mask.unsqueeze(1) * ftr_change.unsqueeze(0).unsqueeze(-1)
# where
#   locus_mask [nbatch x nrole] is a binary matrix s.t.
#       locus_mask[i,k] = 1 iff apply rule at kth position in ith batch member
#   ftr_change [nftr] = (ftr_new - ftr_old)
class PhonoRule(nn.Module):

    def __init__(self, rule_name=''):
        super(PhonoRule, self).__init__()
        self.rule_name = rule_name
        if rule_name == 'initial_voicing':
            self.forward = self.initial_voicing
        elif rule_name == 'final_a_raising':
            self.forward = self.final_a_raising
        else:
            print('PhonoRule: ' + rule_name + ' not registered')
            sys.exit(0)

    def initial_voicing(self, form):
        """
        Stem-initial obstruent voicing (hypothetical)
        """
        sonorant_indx = config.ftrs.index('sonorant')
        voice_indx = config.ftrs.index('voice')
        obstruent = hardtanh(-1.0 * form[:, sonorant_indx, 1], 0.0, 1.0)
        voiceless = hardtanh(-1.0 * form[:, voice_indx, 1], 0.0, 1.0)
        match = obstruent * voiceless
        # Change loci
        mask = torch.zeros_like(form, requires_grad=False, device=config.device)
        mask[:, voice_indx, 1] = 1.0
        mask = mask * match.unsqueeze(-1).unsqueeze(-1)
        #print(torch.min(mask), torch.max(mask)); sys.exit(0)
        # Apply change (note -2x + x = -x)
        change = -2.0 * mask * form
        form_ = form + change
        return form_

    def final_a_raising(self, form):
        """
        Word-final raising a -> ɔ (Javanese)
        """
        nbatch, dsym = form.shape[0], form.shape[1]

        # Locus of application
        # xxx assumes all inputs end in /a/ !
        word_final = hardtanh(form[:, 2, :], 0.0, 1.0)
        word_final = torch.cat(
            [word_final[:, 1:],
             torch.zeros((nbatch, 1), device=config.device)], 1)
        word_final = word_final.unsqueeze(1)  # same locus for all features

        # Featural change
        ftr_A = config.F[:, config.syms.index('a')]
        ftr_O = config.F[:, config.syms.index('ɔ')]
        ftr_change = (ftr_O - ftr_A)
        dftr = ftr_change.shape[0]
        if dftr < dsym:  # append zeros in correspondence index slots
            ftr_change = torch.cat(
                [ftr_change,
                 torch.zeros(dsym - dftr, device=config.device)], 0)
        ftr_change = ftr_change.unsqueeze(0).unsqueeze(
            -1)  # same change for all batch, posns

        # Rule application
        #print(form.shape, word_final.shape, ftr_change.shape)
        form_ = form + word_final * ftr_change
        return form_
