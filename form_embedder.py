#!/usr/bin/env python
# -*- coding: utf-8 -*-
# todo
# - allow segment embeddings to be refined in training (?except for sym, begin, end features)

from .environ import config
from .segment_embedder import SegmentEmbedder
from .role_embedder import RoleEmbedder
from .distance import euclid_squared
import torch
#import tpr
#from tpr import *
import re, sys


class FormEmbedder():
    def __init__(self, symbol_params, role_params):
        self.segment_embedder = SegmentEmbedder(**symbol_params)
        self.role_embedder = RoleEmbedder(**role_params)
        self.sym2id = { sym:i for i,sym in enumerate(self.segment_embedder.syms) }
        self.id2sym = { i:sym for sym,i in self.sym2id.items() }
        print (self.sym2id)
        print (self.id2sym)

    def sym2vec(self, x):
        # Map symbol to filler vector
        F = config.F
        return F.data[:, self.sym2id[x]]
    
    def string2vec(self, x, delim=True):
        # Map space-separated string to matrix of filler vectors
        F = config.F
        idx, lens = self.string2idvec(x, delim)
        return F[:,idx], lens

    def string2idvec(self, x, delim=True, pad=False):
        # Map space-separated string to vector of indices (torch.LongTensor), 
        # possibly with zero-padding at end; also return string length
        y = string2delim(x) if delim else x
        y_idx = [self.sym2id[yi] if yi in self.sym2id else None
                    for yi in y.split(' ')]
        if None in y_idx:
            print ('string2idvec error:')
            print ([sym_idx for sym_idx in zip(y.split(' '), y_idx)])
            print (self.sym2id)
        y = y_idx
        y_len = torch.LongTensor([len(y),])
        if pad:
            y = y + [0,]*(config.nrole - len(y))
        y = torch.LongTensor(y)
        return y, y_len

    def string2tpr(self, x, delim=True):
        # Map space-separated string to tpr
        y, y_len = self.string2idvec(x, delim)
        if y_len > config.nrole:
            print('string2tpr error: length of string (={}) longer than nrole (={}) for input: {}'.format(y_len, config.nrole, x))
            return None
        Y = torch.zeros(config.dfill, config.drole)
        for i in range(y_len):
            Y += torch.ger(config.F[:,y[i]], config.R[:,i]) # outer product
        return Y

    def idvec2string(self, x, copy=None, pivot=None, trim=True):
        # Map idvec to string, marking up output with deletion 
        # and (un)pivot flags
        segs = [self.id2sym[idx] for idx in x]
        y = ' '.join(segs)
        if trim:
            y = re.sub('(?<='+ config.stem_end +').*', '', y)
        segs = y.split(' ')
        if copy is not None:
            segs = ['⟨'+x+'⟩' if (i<len(copy) and copy[i]<0.5) else x for i,x in enumerate(segs)]
        if pivot is not None:
            segs = [x+' •' if (i<len(pivot) and pivot[i]>0.5) else x for i,x in enumerate(segs)]
        y = ' '.join(segs)
        return y
    
    def tpr2string(self, X, trim=True):
        # Map tpr to string by comparing successive unbound vectors 
        # to pure filler vectors with squared euclidean distance
        segs = []
        for j in range(config.nrole):
            y = X @ config.U[:,j]   # unbind
            y_dist = euclid_squared(y, config.F)
            y_idx = torch.argmin(y_dist).item()
            segs.append(self.id2sym[y_idx])
        y = ' '.join(segs)
        if trim:
            y = re.sub('(?<='+ config.stem_end +').*', '', y)
        return y


# todo: make the following class-level utility functions
def string2sep(x):
    # Separate characters of string with spaces
    y = ' '.join([xi for xi in x])
    return y

def string2delim(x):
    # Add word delimiters; input must be space-separated
    y = [config.stem_begin,] + [xi for xi in x.split(' ')] + [config.stem_end,]
    y = ' '.join(y)
    return y

def string2undelim(x):
    # Remove word delimiters; input can be space-separated
    y = re.sub('^'+ config.stem_begin +'[ ]*', '', x)
    y = re.sub('[ ]*'+ config.stem_end +'$', '', y)
    return y
