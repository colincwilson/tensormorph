#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import collections
import numpy as np
import re, sys

import tpr
from tpr import *
from seq_embedder import SeqEmbedder, string2sep, string2delim
from morph_embedder import MorphEmbedder

DataPoint = collections.namedtuple(\
    'DataPoint',\
    ['stem', 'morph', 'targ', 'Stem', 'Morph', 'Targ', 'targ_len']\
)

DataBatch = collections.namedtuple(\
    'DataBatch',\
    ['stems', 'morphs', 'targs', 'Stems', 'Morphs', 'Targs', 'targ_lens']\
)

class DataSet():
    def __init__(self, dat, held_in_stems=None, held_out_stems=None, vowels=None):
        self.dat            = dat
        self.held_in_stems  = held_in_stems
        self.held_out_stems = held_out_stems
        self.vowels         = vowels
        self.train          = None
        self.test           = None
        self.train_embed    = None
        self.test_embed     = None

        segments = set() if vowels is None else set(vowels)
        for form in dat['output']:
            segments |= set(form.split(' '))
        segments = [x for x in segments]
        segments.sort()
        self.segments = segments


    # get subset defined by morphological tag
    def subset(self, morph):
        dat     = self.dat
        dat_sub = dat[dat['morph'] == morph]
        return DataSet(dat_sub, vowels=self.vowels)


    # split into train and test subsets
    def split(self, test_size=0.25):
        dat = self.dat.copy()
        held_in_stems = self.held_in_stems
        held_out_stems = self.held_out_stems

        # remove held_in and held_out prior to random split
        if held_in_stems is not None:
            held_in = dat[(dat.stem.isin(held_in_stems))]
            dat      = dat[~(dat.stem.isin(held_in_stems))]
            dat.reset_index()
        if held_out_stems is not None:
            held_out = dat[(dat.stem.isin(held_out_stems))]
            dat      = dat[~(dat.stem.isin(held_out_stems))]
            dat.reset_index()

        train, test = train_test_split(dat, test_size=test_size)

        if held_in_stems is not None:
            train = pd.concat([held_in, train])
            train.reset_index()
        if held_out_stems is not None:
            test = pd.concat([held_out, test])
            test.reset_index()

        print(train.head())
        print(test.head())
        self.train, self.test = train, test
        return None


    # embed one example
    def embed1(self, ex):
        stem, morph, targ = ex['stem'], ex['morph'], ex['output']
        stem = string2delim(stem)
        targ = string2delim(targ)

        try:    Stem = tpr.seq_embedder.string2tpr(stem, False)
        except: print ('error embedding stem', stem)
        try:    Morph = tpr.morph_embedder.embed(morph)
        except: print ('error embedding morph', morph)
        try:    Targ, targ_len = tpr.seq_embedder.string2idvec(targ, False)
        except: print ('error embedding target', targ)

        Stem = Stem.unsqueeze(0)
        Morph = Morph.unsqueeze(0)
        Targ = Targ.unsqueeze(0)
        return DataPoint(stem, morph, targ, Stem, Morph, Targ, targ_len)


    # embed training and testing examples
    def embed(self):
        self.train_embed = [self.embed1(ex)\
            for i,ex in self.train.iterrows()]
        self.test_embed  = [self.embed1(ex)\
            for i,ex in self.test.iterrows()]
        return None


    # get a random batch of training examples
    def get_batch(self, nbatch=20, start_index=0):
        train_embed = self.train_embed
        n       = len(train_embed)
        indx    = np.random.choice(n-start_index, nbatch, replace=False)
        indx    = [x+start_index for x in indx]

        train_batch = [train_embed[i] for i in indx]
        stems       = [ex.stem for ex in train_batch]
        morphs      = [ex.morph for ex in train_batch]
        targs       = [ex.targ for ex in train_batch]
        Stems       = torch.cat([ex.Stem for ex in train_batch], 0)
        Morphs      = torch.cat([ex.Morph for ex in train_batch], 0)
        Targs       = torch.cat([ex.Targ for ex in train_batch], 0)
        targ_lens   = [ex.targ_len for ex in train_batch]
        return DataBatch(stems, morphs, targs, Stems, Morphs, Targs, targ_lens)
