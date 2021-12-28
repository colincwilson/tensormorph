# -*- coding: utf-8 -*-

import config
from tpr import *
from morph import MorphOp, Morph
from cogrammar import Cogrammar
from birnn_pivoter import BiRNNPivoter
#from phonology import PhonoRules, PhonoRule
from correspondence import BiCorrespondence
from affixer import AffixVocab  # xxx spurious


class RedupCogrammar(nn.Module):

    def __init__(self):
        super(RedupCogrammar, self).__init__()
        self.base_cogrammar = Cogrammar()
        self.red_cogrammar = Cogrammar()
        self.pivoter = BiRNNPivoter()  # xxx replace
        self.unpivoter = BiRNNPivoter()  # xxx replace
        self.morph_op = MorphOp()
        #self.combiner = Combiner()
        #self.phono_rules = PhonoRule('final_a_raising') #PhonoRules(config.dmorphosyn+2)
        self.correspondence = BiCorrespondence()
        self.reduplication = True  # xxx needed by recorder
        self.affixer = Affixer(1, False)  # xxx needed by recorder

        # xxx test init
        #self.pivoter.init(ftr='end', before=True, bias=10.0, clamp=True)
        #self.unpivoter.init(ftr='end', before=True, bias=10.0, clamp=True)

    def forward(self, stem, morphosyn, max_len=10):
        """
        Apply base and red cogrammars to stem, 
        combine results into output
        """
        # Append correspondence indices to stem
        #stem = self.correspondence.indexify(stem)
        #Stem_indexed = Stem.clone().detach()

        # Apply base and red cogrammars to stem
        #base = self.phono_rules(stem)
        base = self.base_cogrammar(
            Morph(stem.form, stem.form_str), morphosyn, max_len)
        red = self.red_cogrammar(
            Morph(stem.form, stem.form_str), morphosyn, max_len)

        # Determine position of reduplicant insertion
        # in base and end of reduplicant
        context = morphosyn  # xxx no morphophonological conditioning
        base_pivot = self.pivoter(base.form, context)[0]
        red_pivot = self.unpivoter(red.form, context)[0]

        # Copy all symbols of base and reduplicant (i.e.,
        # all deletion must be done in base/red cogrammars)
        base_copy = torch.ones(stem.form.shape[0], config.nrole)
        red_copy = torch.ones(stem.form.shape[0], config.nrole)

        # Combine base and reduplicant into output tpr
        # note: make new morphs to separate outputs of base/red
        # cogrammars from inputs to root cogrammar
        base = Morph(base.form, pivot=base_pivot, copy=base_copy)
        red = Morph(red.form, pivot=red_pivot, copy=red_copy)
        output = self.morph_op(base, red)

        # Recopy/backcopy within output
        #output = self.correspondence(output, max_len)

        # Remove correspondence indices
        #output = self.correspondence.deindexify(output)

        if config.recorder is not None:
            self.stem = base
            self.affix = red
            self.output = output
        return output

    def init(self):
        pass