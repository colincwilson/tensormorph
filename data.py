#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re, sys

from .environ import config
from .tpr import *
from .seq_embedder import string2sep, string2delim, string2undelim
#from morph_embedder import MorphEmbedder

# todo: change to mutable recordtype
DataPoint = collections.namedtuple(
    'DataPoint',
    ['stem', 'morph', 'output', 'Stem', 'Morph', 'Output', 'output_len', 'pred'],
    defaults = (None,)*8
)

# todo: change to mutable recordtype
DataBatch = collections.namedtuple(
    'DataBatch',
    ['stems', 'morphs', 'outputs', 'Stems', 'Morphs', 'Outputs', 'output_lens', 'max_len', 'preds'],
    defaults = (None,)*9
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
        for form in dat['stem']:
            segments |= set(form.split(' '))
        for form in dat['output']:
            segments |= set(form.split(' '))
        segments = [x for x in segments]
        segments.sort()
        self.segments = segments


    # get subset defined by morphological tag, 
    # stem regex, and output regex
    def subset(self, morph=None, stem_regex=None, output_regex=None):
        dat1    = self.dat
        if morph is not None:
            dat1    = dat1[dat1['morph'] == morph]
        #print (dat1.head())
        if stem_regex is not None:
            dat1 = dat1[dat1.stem.str.match(stem_regex)]
            #print (stem_regex, len(dat1))
        if output_regex is not None:
            dat1 = dat1[dat1.output.str.match(output_regex)]
            #print (output_regex, len(dat1))
        return DataSet(dat1, vowels=self.vowels)


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


     # embed training and testing examples
    def embed(self):
        self.train_embed = [self.embed1(ex)\
            for i,ex in self.train.iterrows()]
        self.test_embed  = [self.embed1(ex)\
            for i,ex in self.test.iterrows()]
        return None


   # embed one example
    def embed1(self, ex):
        stem, morph, output = ex['stem'], ex['morph'], ex['output']
        stem    = string2delim(stem)
        output  = string2delim(output)

        seq_embedder    = config.seq_embedder
        morph_embedder  = config.morph_embedder
        try:    Stem = seq_embedder.string2tpr(stem, False)
        except: print ('error embedding stem', stem)
        try:    Morph = morph_embedder.embed(morph)
        except: print ('error embedding morph', morph)
        try:    Output, output_len = seq_embedder.string2idvec(output, False)
        except: print ('error embedding output', output)

        Stem    = Stem.unsqueeze(0)
        Morph   = Morph.unsqueeze(0)
        Output  = Output.unsqueeze(0)
        return DataPoint(stem, morph, output, Stem, Morph, Output, output_len)


    # get a random batch of training examples, or all training examples 
    # (unrandomized), or all testing examples (unrandomized)
    def get_batch(self, type='train_rand', nbatch=20, start_index=0):
        if type=='train_rand':
            train_embed = self.train_embed
            n           = len(train_embed)
            indx        = np.random.choice(n-start_index, nbatch, replace=False)
            batch       = [train_embed[i] for i in indx]
        elif type=='train_all':
            batch       = self.train_embed
        elif type=='test_all':
            batch       = self.test_embed

        stems       = [ex.stem for ex in batch]
        morphs      = [ex.morph for ex in batch]
        outputs     = [ex.output for ex in batch]
        Stems       = torch.cat([ex.Stem for ex in batch], 0)
        Morphs      = torch.cat([ex.Morph for ex in batch], 0)
        Outputs     = torch.cat([ex.Output for ex in batch], 0)
        output_lens = [ex.output_len for ex in batch]
        max_len     = np.max(output_lens)
        return DataBatch(
            stems, morphs, outputs,
            Stems, Morphs, Outputs,
            output_lens, max_len)


# write batch after predicting outputs
# xxx relocate? xxx use string2undelim?
def write_batch(batch, fname):
    batch_dump = pd.DataFrame({
        'stem':     [stem for stem in batch.stems],
        'output':   [output for output in batch.outputs],
        'morph':    [morph for morph in batch.morphs],
        'pred':     [pred for pred in batch.preds]
    })
    batch_dump.to_csv(fname, index=False)