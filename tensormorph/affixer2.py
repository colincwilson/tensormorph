# -*- coding: utf-8 -*-

import config
from tpr import *
from morph import Morph
from scanner import propagate
from position_pivoter import PositionPivoter


class Affixer2(nn.Module):
    """
    Vocabulary items as in Distributed Morphology
    """

    def __init__(self, morphophon=False):
        super(Affixer2, self).__init__()
        self.naffix = naffix = 50  # xxx config option
        self.pivoter = pivoter = PositionPivoter()

        # Affix-specific specifications
        self.affix_context = \
            Parameter(0.1 * torch.randn(naffix, config.dcontext))
        self.affix_context.data[:config.dcontext, :].diagonal().fill_(3.0)
        self.affix_form = \
            Parameter(0.1 * torch.randn(naffix, config.dsym * config.nrole))
        self.affix_end = \
            Parameter(0.1 * torch.randn(naffix, config.nrole))
        self.affix_copy = \
            Parameter(0.1 * torch.randn(naffix, config.nrole))
        self.affix_Wpivot = \
            Parameter(0.1 * torch.randn(naffix, pivoter.npivot)) # pivot logits

        # Bias terms (~ prior on affixes)
        self.affix_form0 = \
            Parameter(0.1 * torch.randn(1, config.dsym * config.nrole))
        self.affix_end0 = Parameter(0.1 * torch.randn(1, config.nrole))
        self.affix_copy0 = Parameter(0.1 * torch.randn(1, config.nrole))
        self.affix_Wpivot0 = Parameter(0.1 * torch.randn(1, pivoter.npivot))

        # Explicit zero affixation
        self.zero_affix = nn.Linear(config.dcontext, 1)

        # Softmax precision over affix vocab
        self.tau = Parameter(0.1 * torch.randn(1))

        # Rule weights and self-inhibition
        #self.alpha = Parameter(0.1 * torch.randn(1, naffix))
        #self.beta = Parameter(0.1 * torch.randn(1, naffix))
        #self.beta_vec = None

        # Dimension weights
        #self.wdim = Parameter(0.1 * torch.randn(config.ndim + 1))

    #def reset(self, nbatch):
    #    self.beta_vec = torch.zeros(nbatch, self.naffix, requires_grad=True)

    def forward(self, Stem, context):
        nbatch = Stem.form.shape[0]
        tau = exp(self.tau)
        #alpha = exp(self.alpha)
        #beta = exp(self.beta)
        #wdim = exp(self.wdim)

        # Similarity of specified context to each affix context
        #context = [wdim[i].unsqueeze(0) * context[i] # xxx context weights
        #           for i in range(len(context))]
        context = torch.cat(context, -1)
        affix_context = sigmoid(self.affix_context)  # tanh
        # xxx exp(self.affix_context)
        sim = einsum('bc,ac->ba', context, affix_context)
        # Logits [nbatch x naffix]

        #sim = sim - beta * self.beta_vec # activation - inhibition

        # Softmax over affix lexicon
        #sim = sim / float(config.dcontext)**0.5 # xxx scaled dot product
        act = softmax(tau * sim, dim=-1)  # Activations [nbatch x naffix]
        #print(f'act: {act[0]}')

        #self.beta_vec = self.beta_vec + act # xxx accumulate rule use

        # Affix form
        affix_form = self.affix_form + self.affix_form0  # TPRs [naffix x (dsym * nrole)]
        affix_form = tanh(affix_form)
        affix_form = einsum('ba,ax->bx', act, affix_form)  # act @ affix_form
        affix_form = affix_form.view(nbatch, config.dsym, config.nrole)
        affix_form.data[:, :, 10:] = 0.0  # xxx max affix length
        #affix_form.data[:,1,:] = 0.0 # xxx no word boundaries in affixes,
        # otherwise model uses truncation-by-affixation, placing affix
        # with end-delimiter inside the stem of affixation

        # Affix end/unpivot
        affix_end = self.affix_end + self.affix_end0  # Logits [naffix x nrole]
        affix_end.data[:, 10:] = -10.0  # xxx max affix length
        affix_end = softmax(affix_end, dim=-1)
        #affix_end.data[:,10:] = 10.0 # xxx max affix length
        #affix_end = sigmoid(affix_end)
        affix_end = einsum('ba,ar->br', act, affix_end)  # act @ affix_end
        affix_unpivot = affix_end

        # String well-formedness: soft-enforce epsilons after unpivot
        mask = 1.0 - propagate(affix_unpivot, inclusive=0, direction='LR->')
        #mask = mask.detach() # xxx block gradient?
        #delim_mask = 1.0 - propagate(hardtanh0(-affix_form[:,1,:]),
        #                inclusive = 0, direction = 'LR->')
        affix_form = affix_form * mask.unsqueeze(1)  # broadcast over ftrs

        # Affix copy
        affix_copy = self.affix_copy + self.affix_copy0  # Logits [naffix x nrole]
        affix_copy.data[:, 10:] = -10.0  # xxx max affix length
        affix_copy = sigmoid(affix_copy)
        affix_copy = einsum('ba,ar->br', act, affix_copy)  # act @ affix_copy

        # Affix morph
        Affix = Morph(
            form=affix_form,  # local2distrib(affix_form)
            pivot=affix_unpivot,
            copy=affix_copy)

        # Stem copy
        Stem.copy = torch.ones(
            nbatch, config.nrole,
            device=config.device)  # xxx enforce full stem copy

        # Stem pivot
        W = self.affix_Wpivot + self.affix_Wpivot0  # [naffix x nrole]
        assert (not np.any(np.isnan(W.cpu().data.numpy()))), \
            print(f'W is nan {self.affix_Wpivot}, {self.affix_Wpivot0}')
        W = softmax(W, dim=-1)
        W = einsum('ba,ap->bp', act, W)  # act @ W
        # [nbatch x npivot] xxx [naffix x npivot]
        #stem_pivot = self.pivoter(Stem, W) # [nbatch x naffix x nrole]
        #stem_pivot = einsum('ba,bar->br', act, stem_pivot)
        stem_pivots = self.pivoter(Stem)  # [nbatch x npivot x nrole]
        #print(W.shape, stem_pivots.shape)
        stem_pivot = einsum('bp,bpr->br', W, stem_pivots)
        Stem.pivot = stem_pivot

        # Zero affixation
        zero_affix = sigmoid(self.zero_affix(context)).unsqueeze(-1)

        return Stem, Affix, zero_affix
