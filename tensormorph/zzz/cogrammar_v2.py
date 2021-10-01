#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
from tpr import *
from affixer import Affixer
#from stem_modifier import StemModifier
from truncater import BiTruncater
from combiner import Combiner
from phonology import PhonoRules, PhonoRule

class Cogrammar(nn.Module):
    """
    Cogrammar that performs stem modification and affixation
    """
    def __init__(self):
        super(Cogrammar, self).__init__()
        self.reduplication = False # xxx remove
        self.correspondence = None # xxx remove

        self.affixer = Affixer(
            morphophon = False)

        #self.stem_modifier = StemModifier(
        #    dcontext = config.dmorphosyn)
        self.truncater = BiTruncater(1, 5)

        self.combiner = Combiner()
        #self.combiner.morph_attender.tau.data.fill_(3.0) # xxx
        #self.combiner.posn_attender.tau.data.fill_(3.0)
        #self.combiner.morph_attender.tau.requires_grad = False
        #self.combiner.posn_attender.tau.requires_grad = False

        # xxx test initialization
        #self.affixer.init(bias=2.0, clamp=False)
        #self.truncater.init(bias=2.0, clamp=False)


    def forward(self, Stem, Morphosyn, max_len):
        """
        Map stem to modified and affixed stem
        """
        nbatch  = Stem.shape[0]

        Affix, copy_affix, pivot, unpivot = \
            self.affixer(Stem, Morphosyn)

        copy_stem = self.truncater(Stem, Morphosyn)
        #copy_stem = torch.ones(nbatch, config.nrole)

        # Combine stem and affix into output tpr
        Output  = self.combiner(Stem, Affix,
                                copy_stem, copy_affix, 
                                pivot, unpivot, max_len)

        # xxx todo: reactivate phonology
        #output = self.phono_rules(stem, context)
        #output = self.phono_rules(output, context)

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'Stem': Stem,
                'Affix': Affix,
                'Output': Output,
                'copy_stem': copy_stem,
                'copy_affix': copy_affix, 
                'pivot': pivot,
                'unpivot': unpivot
                })

        #return Output, Affix, (copy_stem, copy_affix, pivot, unpivot)
        return Output