# -*- coding: utf-8 -*-

# see also /Users/colin/Dropbox/TensorProductStringToStringMapping/00R/startEndRole.R for Henderson (1998) start-end roles
# see also OSCAR: Brown, G. D., Preece, T., & Hulme, C. (2000). Oscillator-based memory for serial order. Psychological review, 107(1), 127.
# xxx assumes localist roles
# todo: handle correspondence indices on affixes
#   - rename indexify() as indexify_stem()
#   - add indexify_affix() which uses opposite-sign indices
#       (note: this is correct for only one affix)
#   - alternative (more general): include morpheme identifier as
#       component of index, as in McCarthy 1979, et seq.;
#       need this information anyway ...
# todo: replace LR-> and <-RL directions with stem (base) vs. affix (reduplicant)

import config
from tpr import *
import numpy as np


class BiCorrespondence(nn.Module):
    """
    Correspondence relation that acts bidirectionally
    """

    def __init__(self, index_type='onehot', nindex=None):
        super(BiCorrespondence, self).__init__()
        # Correspondence indices
        self.nindex = nindex \
            if index_type!='onehot' and nindex is not None \
            else config.nrole + 1 # +1 to aid debugging
        if index_type == 'onehot':
            self.indices = torch.eye(
                self.nindex, requires_grad=False)[:, 0:config.nrole]
        elif index_type == 'sinusoid':
            print('correspondence: sinusoid indices not implemented')
            sys.exit(0)

        # Correspondence relation that acts LR-> (i.e., progressively)
        self.correspondence_LR = Correspondence(
            nindex=self.nindex, direction='LR->')
        # Correspondence relation that acts <-RL (i.e., regressively)
        self.correspondence_RL = Correspondence(
            nindex=self.nindex, direction='<-RL')
        # Weighting of LR-> and <-RL correspondence relations
        # xxx should allow to apply in both directions?
        self.alpha = Parameter(torch.randn(1))

    # Copy corresponding segments bidirectionally (weighted)
    def forward(self, output, max_len):
        alpha = sigmoid(self.alpha)
        output_LR = self.correspondence_LR(output, max_len)
        output_RL = self.correspondence_RL(output, max_len)
        output_new = alpha * output_LR + \
                     (1.0 - alpha) * output_RL

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'alpha': alpha,
                'indices': self.indices
            })

        return output_new

    # Append correspondence indices to bottom of tprs
    # todo: mask out indices for unfilled roles?
    def indexify(self, forms):
        nbatch = forms.shape[0]
        nindex, nrole = self.indices.shape
        indices = self.indices.unsqueeze(0) \
                    .expand(nbatch, nindex, nrole)
        #mask = torch.narrow(X, 1, 0, 1)
        #indices = indices * mask
        forms_indexed = torch.cat([forms, indices], 1)
        return forms_indexed

    # Append null (xxx or negative?) correspondence
    # indices to bottom of tprs (xxx for fixed affixes only)
    def indexify_null(self, forms):
        nbatch = forms.shape[0]
        nindex, nrole = self.indices.shape
        #indices = -self.indices.unsqueeze(0) \
        #            .expand(nbatch, nindex, nrole)
        indices = torch.zeros((nbatch, nindex, nrole), requires_grad=False)
        # xxx testing
        forms_indexed = torch.cat([forms, indices], 1)
        return forms_indexed

    # Strip correspondence indices from bottom of tprs
    # note: makes no change if indices are not present
    def deindexify(self, forms):
        forms_deindexed = torch.narrow(forms, 1, 0, config.dsym)
        return forms_deindexed


class Correspondence(nn.Module):
    """
    Correspondence relation that acts unidirectionally (LR-> or <-RL)
    xxx assumes localist roles
    """

    def __init__(self, nindex, direction):
        super(Correspondence, self).__init__()
        self.nindex = nindex
        self.direction = direction

        # Precision of self-attention (higher -> more precise)
        self.tau = Parameter(torch.randn(1))
        # Strength of faithfulness (higher -> more faithfulness)
        self.beta = Parameter(torch.randn(1))

    # Copy corresponding segments in specified direction
    def forward(self, output, max_len):
        tau = exp(self.tau)
        beta = self.beta

        # Access correspondence indices (nbatch x nindex x nrole), place
        # indices in 'rows' by transposing to (nbatch x nrole) x nindex
        indices = torch.narrow(output, 1, config.dsym, self.nindex)
        indices = indices.transpose(2, 1)
        #print ('indices:', indices.shape)

        # Compute similarities for all index pairs
        # i.e., [nbatch x 1 x nrole x nindex] - [nbatch x nrole x 1 x nindex]
        # then sum squared differences, followed by multiplicative inverse
        # xxx relocate to distance.py
        dist = indices.unsqueeze(1) - indices.unsqueeze(2)
        dist = torch.sum(dist**2.0, -1)  # xxx mean instead of sum?
        sim = -tau * dist
        #print('dist:', dist.shape, 'sim:', sim.shape)

        # Mask similarities depending on direction
        # (retain values for which mask is 1 = True)
        # see also torch.nn.modules.transformer.generate_square_subsequent_mask
        mask = torch.ones_like(sim, dtype=torch.uint8, requires_grad=False)
        if self.direction == 'LR->':
            # retain entry [i,j] if r_i precedes r_j
            mask = torch.triu(mask, diagonal=1)
        elif self.direction == '<-RL':
            # retain entry [i,j] if r_i follows r_j
            mask = torch.tril(mask, diagonal=-1)
        else:
            print('correspondence: unknown direction', self.direction)
            sys.exit(0)
        sim = sim.masked_fill(mask == 0, -1.0e10)
        #print (sim.shape)

        # Normalize exp similarities for each target position
        # todo: avoid redundant computation of softmax denominator
        attn = softmax(sim, 1)  # softmaxed similarities
        logZ = sim.logsumexp(1, keepdim=False)  # log softmax denominator
        Z = exp(logZ)  # softmax denominator
        # print ('attn', attn)
        # print ('logZ:', logZ[0], 'Z:', Z[0])

        # Faithful update for each role
        faith = output @ attn  #torch.matmul(output, attn)
        # print('faith:', faith)

        # Faithfulness probabilities [nbatch x nrole]
        alpha = Z / (Z + exp(-beta))
        #print('exp(tau):', tau)
        #print('max(alpha):', torch.max(alpha))
        #print('alpha:', alpha.detach().numpy())

        # Apply updates (affects all features for each nbatch x nrole)
        update = alpha.unsqueeze(1) * (faith - output)
        output_new = output + update

        self.trace = {
            'sim': sim,
            'attn': attn,
            'alpha': alpha,
            'update': update
        }
        return output_new