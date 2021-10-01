#!/usr/bin/env python
# -*- coding: utf-8 -*-

# xxx replaces vocab_inserter
# xxx todo: add no-op initialization

from environ import config
from tpr import *
#from scanner import BiScanner, BiLSTMScanner
import pivoter
from pivoter import *

class Affixer(nn.Module):
    """
    Retrieve form of affix, pivoter, unpivot, etc. from morphosyn (and stem)
    """
    def __init__(self, morphophon=False):
        super(Affixer, self).__init__()
        self.morphophon = morphophon
        if morphophon:
            # Stem is scanned for morphophonological info
            self.stem_scanner = BiLSTMScanner(
                hidden_size = 1)
            dcontext = config.dmorphosyn + 2
        else:
            self.stem_scanner = None
            dcontext = config.dmorphosyn
        self.dcontext = dcontext

        self.context2affix = nn.Linear(
            dcontext,
            config.dfill * config.drole,
            bias=True)
        #dhidden = 10
        #self.context2affix = nn.Sequential(
        #    nn.Linear(dcontext, dhidden, bias=True),
        #    nn.ReLU(),
        #    nn.Linear(dhidden, config.dfill * config.drole, bias=True)
        #)

        self.context2affix_copy = nn.Linear(
            dcontext,
            config.nrole,
            bias=True)
        
        self.pivoter = BiPivoter(dcontext, nfeature=5)

        self.context2unpivot = nn.Linear(
            dcontext,
            config.nrole,
            bias=True)

    def forward(self, Stem, Morphosyn):
        nbatch = Stem.shape[0]

        if self.morphophon:
            Morphophon = self.stem_scanner(Stem)
            Context = torch.cat([Morphosyn, Morphophon], 1)
        else:
            Context = Morphosyn
        
        Affix = self.context2affix(Context) \
                    .view(nbatch, config.dfill, config.drole) # xxx config.nrole
        Affix = sigmoid(Affix) if config.privative_ftrs \
                    else tanh(Affix)
        Affix.data[:,0].fill_(1.0) # clamp non-epsilon on
        # xxx replace sigmoid | tanh with linear?
        
        pivot, _, _ = self.pivoter(Stem, Context)

        unpivot0 = torch.arange(config.nrole) # xxx unpivot biases
        unpivot = self.context2unpivot(Context)
        unpivot = sigmoid(unpivot + unpivot0) \
                    .view(nbatch, config.nrole) # xxx testing
        # xxx fix pivot at end delimiter
        #end_unpivoter = BiPivoter(self.dcontext, nfeature=5)
        #end_unpivoter.fix(direction='LR->', ftr='end', before=1)
        #unpivot, _, _ = end_unpivoter(Affix, Context)

        affix_copy = self.context2affix_copy(Context)
        affix_copy = sigmoid(affix_copy) \
                        .view(nbatch, config.nrole)
        affix_copy = torch.ones_like(affix_copy) # xxx testing

        return Affix, affix_copy, pivot, unpivot # xxx dict

    def init(self, affix=None, bias=10.0, clamp=True):
        for m in [self.context2affix, self.context2unpivot,
                  self.context2affix_copy]:
            for p in m.parameters():
                p.data.fill_(0.0)
                p.requires_grad = (not clamp)
        if affix is not None: # space-delimited symbol sequence
            affix_len = len(affix.split())
            Affix = config.form_embedder.string2tpr(affix, delim=False)
            self.context2affix.bias.data = (bias * Affix).view(-1)
            self.context2unpivot.bias.data[:affix_len].fill_(-bias)
            self.context2unpivot.bias.data[affix_len:].fill_(bias)
            self.context2affix_copy.bias.data[0:affix_len].fill_(bias)
            self.context2affix_copy.bias.data[affix_len:].fill_(-bias)
        self.pivoter.init()