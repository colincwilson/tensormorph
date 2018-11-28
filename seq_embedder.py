#!/usr/bin/env python
# -*- coding: utf-8 -*-
# todo
# - auto-detect vowels if vowel set not provided
# - allow segment embeddings to be trained (except for delimiter and segment features)

from environ import config
from symbol_embedder import SymbolEmbedder
from role_embedder import RoleEmbedder
import torch
#import tpr
#from tpr import *
import re, sys

class SeqEmbedder():
    def __init__(self, features=None, segments=None, vowels=None, nrole=20, random_roles=False):
        self.symbol_embedder = SymbolEmbedder(features, segments, vowels)
        self.role_embedder = RoleEmbedder(nrole, random_roles)
        self.sym2id = { sym:i for i,sym in enumerate(config.syms) }
        self.id2sym = { i:sym for sym,i in self.sym2id.items() }
        print (self.sym2id)
        print (self.id2sym)

    # get filler vector for symbol
    def sym2vec(self, x):
        F       = config.F
        sym2id  = self.sym2id
        return F.data[:, sym2id[x]]

    # map string to vector of indices
    # (input must be space-separated)
    def string2idvec(self, x, delim=True):
        sym2id  = self.sym2id
        y       = self.string2delim(x) if delim else x
        y       = [sym2id[yi] for yi in y.split(u' ')]
        y_pad   = y + [0,]*(config.nrole - len(y))
        y_pad   = torch.LongTensor(y_pad)
        return y_pad, len(y)

    # map string to tpr
    # input must be space-separated
    def string2tpr(self, x, delim=True):
        sym2id, F, R = self.sym2id, config.F, config.R
        y = string2delim(x) if delim else x
        y = y.split(' ')
        n = len(y)
        if n >= config.nrole:
            print('string2tpr error: string length longer than nrole for string', x)
            return None
        Y = torch.zeros(config.dfill, config.drole)
        for i in range(n):
            try: j = sym2id[y[i]]
            except: print ('string2tpr error: no id for segment', y[i]) 
            Y += torch.ger(F.data[:,j], R.data[:,i]) # outer product
        return Y

    # mark up output with deletion and (un)pivot indicators
    def idvec2string(self, x, copy=None, pivot=None):
        id2sym = self.id2sym
        segs = [id2sym[id] for id in x]
        y = ' '.join(segs)
        y = re.sub(u'⋉.*', u'⋉', y)
        segs = y.split(' ')
        if copy is not None:
            segs = [u'⟨'+x+u'⟩' if (i<len(copy) and copy[i]<0.5) else x for i,x in enumerate(segs)]
        if pivot is not None:
            segs = [x+u' •' if (i<len(pivot) and pivot[i]>0.5) else x for i,x in enumerate(segs)]
        y = ' '.join(segs)
        return y

# separate elements of string with spaces
def string2sep(x):
    x = ' '.join([xi for xi in x])
    return x

# add word delimiters; input must be space-separated
def string2delim(x):
    val = [config.stem_begin,] + [xi for xi in x.split(' ')] + [config.stem_end,]
    val = ' '.join(val)
    return val
