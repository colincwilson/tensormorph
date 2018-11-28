#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
import tpr
from tpr import *
from scanner import BiScanner, BiLSTMScanner
from stem_modifier import StemModifier
from thunker import Thunker
from combiner import Combiner


class Affixer(nn.Module):
    def __init__(self, node='root'):
        super(Affixer, self).__init__()
        self.scanner        = BiLSTMScanner(hidden_size = 1)
        self.pivoter        = BiScanner(morpho_size = config.dmorph+2, nfeature = 5, node = node+'-pivoter')
        self.stem_modifier  = StemModifier()
        if node=='root':
            self.reduplicator = Affixer('reduplicant')
            self.unpivoter  = BiScanner(morpho_size = config.dmorph+2, nfeature = 5, node = node+'-unpivoter')
            self.redup      = Parameter(torch.zeros(1)) # xxx need to modulate by morph
        self.affix_thunker  = Thunker()
        self.combiner       = Combiner()
        self.node           = node


    # map tpr of stem to tpr of stem+affix
    def forward(self, stem, morph, max_len):
        nbatch  = stem.shape[0]
        scan    = torch.zeros((nbatch,2)) # self.scanner(stem)
        morpho  = torch.cat([morph, scan], 1)

        copy_stem = self.stem_modifier(stem, morpho) if 1\
                    else torch.ones((nbatch, config.nrole))
        pivot   = self.pivoter(stem, morpho)
        affix, unpivot, copy_affix = self.get_affix(stem, morpho, max_len)

        output  = self.combiner(stem, affix, copy_stem, copy_affix, pivot, unpivot, max_len)

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'stem_tpr':stem,
                'affix_tpr':affix,
                'copy_stem':copy_stem,
                'copy_affix':copy_affix,
                'pivot':pivot,
                'unpivot':unpivot,
                'output_tpr':output
            })

        return output, affix, (pivot, copy_stem, unpivot, copy_affix)


    def get_affix(self, stem, morpho, max_len):
        if self.node=='root':
            # reduplicative affix
            # xxx fixme
            #affix_redup = self.reduplicator(stem, morpho, max_len)
            #pivot_redup = self.unpivoter(affix0, morpho)
            # non-reduplicative affix
            affix_fixed, pivot_fixed, copy_fixed = self.affix_thunker(morpho)
            # convex combination of two affixes
            #redup = torch.zeros() # sigmoid(self.redup)
            #affix = redup * affix_redup + (1.0 - redup) * affix_fixed
            #pivot = redup * pivot_redup + (1.0 - redup) * affix_fixed
            affix = affix_fixed
            pivot = pivot_fixed
            copy = copy_fixed
        else:
            # enforce non-reduplicative affix
            affix, pivot, copy = self.affix_thunker(morpho)

        return affix, pivot, copy


    def init(self):
        pass
