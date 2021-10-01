# -*- coding: utf-8 -*-

# note: replaces vocab_inserter
# todo: add no-op initialization

import config
from tpr import *
#from scanner import BiScanner, BiLSTMScanner
from morph import Morph
import pivoter
from pivoter import *
from truncater import BiTruncater


class Affixer(nn.Module):
    """
    Retrieve form of affix and pivot, unpivot, etc. from morphosyn 
    (and optionally morphophon properties of stem)
    """
    def __init__(self, dcontext=1, morphophon=False):
        super(Affixer, self).__init__()
        self.context2affix_form = Linear(dcontext,
            config.dfill * config.nrole) # context -> tpr
        self.context2affix_pivot = Linear(dcontext,
            config.nrole) # context -> vec
        self.context2affix_copy = Linear(dcontext,
            config.nrole) # context -> vec
        self.pivoter = \
            BiPivoter(dcontext = 1, 
                      nfeature = 5)
        self.truncater = \
            BiTruncater(dcontext = 1,
                        nfeature = 5)


    def forward(self, stem, context):
        #nbatch = stem.shape[0]
        affix_form = self.context2affix_form(context) \
                        .view_as(stem.form)
        affix_form.data[:,0].clamp(1.0, 1.0)
        affix_pivot = self.context2affix_pivot(context) + \
            torch.arange(config.nrole).unsqueeze(0) # death and taxes, baby
        affix_pivot = sigmoid(affix_pivot)
        affix_copy = sigmoid(self.context2affix_copy(context))
        affix = Morph()
        affix.form = affix_form
        affix.pivot = affix_pivot
        affix.copy = affix_copy

        stem.copy = self.truncater(stem.form, context)
        stem.pivot = self.pivoter(stem.form, context)[0]
        #stem_copy = 5.0 * torch.ones((Stem.shape[0], config.nrole), 
        #    requires_grad=False) # xxx no truncation
        #stem.copy = sigmoid(stem_copy)
        return stem, affix


    # xxx fixme
    def init(self, affix=None, bias=10.0, clamp=True):
        for m in [self.context2affix_form, 
                  self.context2affix_pivot,
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
