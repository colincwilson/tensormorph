#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
from .tpr import *
from .scanner import BiScanner, BiLSTMScanner
from .stem_modifier import StemModifier
from .vocab_inserter import VocabInserter
from .combiner import Combiner
from .phonology import PhonoRules


class Affixer(nn.Module):
    def __init__(self, node='root', reduplication=False):
        super(Affixer, self).__init__()
        self.scanner        = BiLSTMScanner(hidden_size = 1)
        self.pivoter        = BiScanner(morpho_size = config.dmorph+2, nfeature = 5, npattern = 3, node = node+'-pivoter')
        self.stem_modifier  = StemModifier()
        self.combiner       = Combiner()
        self.node           = node
        self.redup          = reduplication
        self.phono_rules    = PhonoRules(config.dmorph+2)

        if node=='root' and reduplication:
            self.reduplicator = Affixer('reduplicant')
            self.unpivoter = BiScanner(morpho_size = config.dmorph+2, nfeature = 5, node = node+'-unpivoter')
        else:
            self.affix_inserter = VocabInserter()


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

        #output = self.phono_rules(stem, morpho)
        #output = self.phono_rules(output, morpho)

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
        if self.redup:
            nbatch = stem.shape[0]
            affix, _, _ = self.reduplicator(stem, morpho.narrow(1,0,1), max_len)
            unpivot = self.unpivoter(affix, morpho)
            copy_affix = torch.ones((nbatch, config.nrole)) # xxx force copy
            #print (affix.shape, unpivot.shape, copy_affix.shape)
        else:
            affix, unpivot, copy_affix = self.affix_inserter(morpho)
        return affix, unpivot, copy_affix


    def init(self):
        pass
