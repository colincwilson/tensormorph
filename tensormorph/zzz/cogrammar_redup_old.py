#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
from tpr import *
from scanner import BiScanner, BiLSTMScanner
from stem_modifier import StemModifier
from vocab_inserter import VocabInserter
from combiner import Combiner
from phonology import PhonoRules, PhonoRule
from correspondence import BiCorrespondence

class Cogrammar(nn.Module):
    def __init__(self, reduplication=False):
        super(Cogrammar, self).__init__()
        self.scanner = \
            BiLSTMScanner(hidden_size = 1)
        self.pivoter = \
            BiScanner(dcontext = 1, #config.dmorph+2,
                      nfeature = 5,
                      npattern = 1)
        self.stem_modifier = \
            StemModifier(dcontext = 1) # config.dmorphosyn+2
        self.combiner = Combiner()
        self.affix_inserter = VocabInserter(dcontext = 1) # config.dmorphosyn+2
        self.phono_rules = PhonoRule('final_a_raising') #PhonoRules(config.dmorphosyn+2)
        self.correspondence = BiCorrespondence()
        self.reduplication = reduplication

        if reduplication:
            self.reduplicator = \
                Cogrammar()
            self.unpivoter = \
                BiScanner(dcontext = 1,
                          nfeature = 5)
            #self.redup_modifier = StemModifier(dcontext = 1,    # xxx unnecessary?
            #                                node = 'redup-modifier') # config.dmorph+2


    def forward(self, stem, morphosyn, max_len=10):
        """
        Map tpr of stem to tpr of affixed stem
        """
        nbatch  = stem.shape[0]

        # Append correspondence indices to stem [optional]
        if self.reduplication:
            stem = self.correspondence.indexify(stem)
        stem_indexed = stem.clone().detach()    # xxx for recording only

        # Scan stem for morphophonological properties, 
        # combine these with morphosyn to make context
        # xxx testing -- reduce morpho to constant 1.0
        scan = torch.zeros((nbatch,2)) # self.scanner(stem)
        context = torch.cat([morphosyn, scan], 1)
        context = context.narrow(1,0,1) # xxx testing

        # Scan stem to determine pivot and copy specs
        # xxx testing -- hard-code initial pivot
        pivot = self.pivoter(stem, context)
        copy_stem = self.stem_modifier(stem, context)

        # Get affix and its unpivot, copy specs
        affix, unpivot, copy_affix = \
            self.get_affix(stem, context, max_len)

        # xxx testing: Apply ad-hoc phono rule to stem of redup
        # xxx should apply after combination, but need 
        # access to stem/affix specifications of output segments
        if self.reduplication:
            stem = self.phono_rules(stem)

        # xxx testing -- enforce reduplicant that is a 
        # a full copy of the stem except that edge symbols 
        # (and epsilons) are marked for deletion; enforce 
        # full copy of stem with prefix pivot
        if not self.reduplication:
            copy_stem = hardtanh(stem[:,0,:]) * \
                        (1.0 - hardtanh(stem[:,1,:], 0, 1)) * \
                        (1.0 - hardtanh(stem[:,2,:], 0, 1))
            pivot = 1.0 - hardtanh(stem[:,0,:]) if 0 \
                    else torch.zeros(nbatch, config.nrole)
            affix = torch.zeros_like(affix)
            unpivot = torch.zeros(nbatch, config.nrole)
            #print(pivot); print(copy_stem)
            #sys.exit(0)
        else:
            pivot = torch.zeros(nbatch, config.nrole)
            pivot[:,0] = 1.0
            copy_stem = torch.ones(nbatch, config.nrole)
                        #hardtanh(stem[:,0,], 0, 1)

        # Combine stem and affix into output tpr
        output  = self.combiner(stem, affix,
                                copy_stem, copy_affix, 
                                pivot, unpivot, max_len)

        # xxx todo: reactivate phono rule bank
        #output = self.phono_rules(stem, context)
        #output = self.phono_rules(output, context)

        # Recopy/backcopy within output [optional]
        # xxx testing: hard-coded direction of application
        if self.reduplication:
            output = self.correspondence(output, max_len)

        # Remove correspondence indices [optional]
        if self.reduplication:
            stem = self.correspondence.deindexify(stem)
            affix = self.correspondence.deindexify(affix)
            output = self.correspondence.deindexify(output)

        if config.recorder is not None:
            if config.correspondence is not None:
                stem_ = self.correspondence.deindexify(stem)
                affix_ = self.correspondence.deindexify(affix)
                output_ = self.correspondence.deindexify(output)
            else:
                stem_, affix_, output_ = stem, affix, output
            config.recorder.set_values(self.node, {
                'stem_tpr': stem_,
                'stem_indexed_tpr':stem_indexed,
                'affix_tpr': affix_,
                'output_tpr': output_,
                'copy_stem': copy_stem,
                'copy_affix': copy_affix, 
                'pivot': pivot,
                'unpivot': unpivot
                })
            # xxx todo: save correspondence indices if reduplication==True

            # xxx testing: dump reduplicant and other info to files
            if 0:
                if not self.reduplication:
                    np.savetxt('/Users/colin/Desktop/redup.txt',
                        np.round(output.data[0,:,:].numpy(), 3), delimiter=',')
                else:
                    np.savetxt('/Users/colin/Desktop/stem.txt',
                        np.round(stem.data[0,:,:].numpy(),3), delimiter=',')
                    np.savetxt('/Users/colin/Desktop/output.txt',
                        np.round(output.data[0,:,:].numpy(),3), delimiter=',')

        # example of dictionary specifying internal tensors to be saved 
        # in recorder xxx todo: send detached values to config.recorder 
        # using set_values() forward hook
        #self.record = {
        #   'stem_tpr': stem_,
        #   'affix_tpr': affix_,
        #   'output_tpr': output_,
        #   'copy_stem': copy_stem,
        #   'copy_affix': copy_affix,
        #   'pivot': pivot,
        #   'unpivot': unpivot 
        # }

        #return output, affix, (pivot, copy_stem, unpivot, copy_affix)
        return output


    def get_affix(self, stem, context, max_len):
        # Non-reduplicative affix
        affix_fixed, unpivot_fixed, copy_fixed = \
            self.affix_inserter(context)

        # Append correspondence indices 
        # to fixed affixes [optional]
        if config.correspondence is not None:
            affix_fixed = self.correspondence.indexify_null(affix_fixed)

        # Reduplicative affix
        if self.reduplication:
            nbatch = stem.shape[0]
            #affix_redup, _, _ = \
            affix_redup = \
                self.reduplicator(stem, context, max_len)
                #self.reduplicator(stem, morphosyn.narrow(1,0,config.dmorphosyn), max_len)
            # xxx testing: unpivot at end of reduplicant, note that 
            # must shift all initial values one position to 
            # the left to get the right semantics for combiner
            unpivot_redup = \
                1.0 - hardtanh(affix_redup[:,0,:], 0, 1)
            unpivot_redup = \
                torch.cat([unpivot_redup[:,1:], torch.ones(nbatch,1)], 1)
                #self.unpivoter(stem, context)
                #self.unpivoter(stem, morphosyn.narrow(1,0,1))
            # xxx always copy all non-epsilon segments of reduplicant
            # (because any deletion could be done by reduplicator)
            copy_redup = \
                hardtanh(affix_redup[:,0,:], 0, 1)
                #self.redup_modifier(stem, context) if 1\
                #    else torch.ones((nbatch, config.nrole))
            
            # Affix selection by morphosyn
            # xxx assume reduplication iff 1.0 in dimension 2 (!)
            ## redup_flag = morphosyn.narrow(1,2,1).unsqueeze(-1) # expand to batch x matrix
            ## affix = redup_flag * affix_redup + (1.0-redup_flag) * affix_fixed
            ## redup_flag = redup_flag.squeeze(-1) # shrink to batch x vector
            ## unpivot = redup_flag * unpivot_redup + (1.0-redup_flag) * unpivot_fixed
            ## copy_affix = redup_flag * copy_redup + (1.0-redup_flag) * copy_fixed
            # xxx testing: force reduplication
            affix = affix_redup
            unpivot = unpivot_redup
            copy_affix = copy_redup

        else:
            affix, unpivot, copy_affix = \
                affix_fixed, unpivot_fixed, copy_fixed
        
        return affix, unpivot, copy_affix


    def init(self):
        pass