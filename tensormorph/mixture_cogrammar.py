# -*- coding: utf-8 -*-

import config
from tpr import *
from morph import MorphOp, Morph
from affixer import AffixVocab
#from stem_modifier import StemModifier
from birnn_pivoter import BiRNNPivoter
from truncater import BiTruncater
#from combiner import Combiner
#from writer import Writer
#from phonology import PhonoRules, PhonoRule
from torch.nn import Parameter, Linear, LSTM, GRU, Sequential


class MixtureCogrammar(nn.Module):
    """
    Cogrammar that performs stem truncation, affixation, ...
    this version explicitly sums over possible pivots
    note: apply with tau_min >= 2.0 for ~ deterministic combination
    """

    def __init__(self):
        super(MixtureCogrammar, self).__init__()
        self.affixer = AffixVocab()
        self.morph_op = MorphOp()  # rename morph_combiner?
        self.reduplication = False
        self.correspondence = None  # xxx not used

        self.alpha = Parameter(torch.randn(2))
        self.beta = Parameter(torch.randn(2))
        self.phi = Parameter(torch.randn(5))

    def forward(self, stem, morphosyn, max_len):
        """
        Map stem to affixed / truncated output
        todo: apply truncation before affixation
        """
        stem, affix = self.affixer(stem, morphosyn)
        _, pivots = self.affixer.pivoter(stem.form, morphosyn)

        nbatch = stem.form.shape[0]
        output = Morph()
        output.reset(nbatch)
        alpha = softmax(self.alpha, -1)
        beta = softmax(self.beta, -1)
        phi = softmax(self.phi, -1)
        print(alpha.data.numpy(), beta.data.numpy(), phi.data.numpy())
        # xxx avoid loops by stacking into single vector
        for drn, pivot_drn in enumerate(pivots):
            for loc, pivot_dir_loc in enumerate(pivot_drn):
                for ftr in range(5):  # pivot feature
                    w = alpha[drn] * beta[loc] * phi[ftr]
                    stem.pivot = pivot_dir_loc[:, ftr]
                    output_i = self.morph_op(stem, affix)
                    output.form = output.form + w * output_i.form

        if config.recorder is not None:
            self.stem = stem
            self.affix = affix
            self.output = output

        return output
