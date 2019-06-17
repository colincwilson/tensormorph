#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re, sys

from .environ import config
from .tpr import *
from .form_embedder import string2sep, string2delim, string2undelim
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
        self.max_len        = 0

        segments = set() if vowels is None else set(vowels)
        for form in dat['stem']:
            segments |= set(form.split(' '))
        for form in dat['output']:
            segments |= set(form.split(' '))
        segments = [x for x in segments]
        segments.sort()
        self.segments = segments
        self.max_len = np.max(
            [len(x.split(' ')) for x in dat['stem']] +
            [len(x.split(' ')) for x in dat['output']]
            )


    def subset(self, morph=None, stem_regex=None, output_regex=None):
        # Extract subset defined by morphological tag, 
        # stem regex, or output regex
        dat1    = self.dat
        if morph is not None:
            dat1 = dat1[dat1['morph'] == morph]
        #print (dat1.head())
        if stem_regex is not None:
            dat1 = dat1[dat1.stem.str.match(stem_regex)]
            #print (stem_regex, len(dat1))
        if output_regex is not None:
            dat1 = dat1[dat1.output.str.match(output_regex)]
            #print (output_regex, len(dat1))
        return DataSet(dat1, vowels=self.vowels)


    def split_and_embed(self, test_size=0.25):
        # Prepare for training
        self.split(test_size)
        self.embed()


    def split(self, test_size=0.25):
        if test_size == 0:
            self.train = self.dat
            self.test = None
            return
        
       # Split into train and test subsets
        dat = self.dat.copy()
        held_in_stems = self.held_in_stems
        held_out_stems = self.held_out_stems

        # Remove held_in and held_out prior to random split
        if held_in_stems is not None:
            held_in = dat[(dat.stem.isin(held_in_stems))]
            dat      = dat[~(dat.stem.isin(held_in_stems))]
            dat.reset_index()
        if held_out_stems is not None:
            held_out = dat[(dat.stem.isin(held_out_stems))]
            dat      = dat[~(dat.stem.isin(held_out_stems))]
            dat.reset_index()

        train, test = train_test_split(dat, test_size=test_size)

        # Combine held_in and held_out with splits
        if held_in_stems is not None:
            train = pd.concat([held_in, train])
            train.reset_index()
        if held_out_stems is not None:
            test = pd.concat([held_out, test])
            test.reset_index()

        print(train.head())
        print(test.head())
        self.train = train
        self.test = test


    def embed(self):
        # Embed training and testing examples
        self.train_embed = [self.embed1(ex)\
            for i,ex in self.train.iterrows()]
        if self.test is not None:
            self.test_embed  = [self.embed1(ex)\
                for i,ex in self.test.iterrows()]
        else:
            self.test_embed = None


    def embed1(self, ex):
       # Embed a single example
        stem, morph, output = ex['stem'], ex['morph'], ex['output']
        stem    = string2delim(stem)
        output  = string2delim(output)

        form_embedder    = config.form_embedder
        morphosyn_embedder  = config.morphosyn_embedder
        try:
            Stem = form_embedder.string2tpr(stem, False)
        except:
            print ('error embedding stem', stem)
        try:
            Morph = morphosyn_embedder.embed(morph)
        except: 
            print ('error embedding morph', morph)
        try:
            Output, output_len = form_embedder.string2idvec(output, delim=False, pad=True) # xxx
        except:
            print ('error embedding output', output)

        Stem    = Stem.unsqueeze(0)
        Morph   = Morph.unsqueeze(0)
        Output  = Output.unsqueeze(0)
        return DataPoint(stem, morph, output, Stem, Morph, Output, output_len)


    def get_batch(self, type='train_rand', nbatch=20, start_index=0):
        # Get a random batch of training examples, or all training examples 
        # (unrandomized), or all testing examples (unrandomized)
        if type=='train_rand':
            train_embed = self.train_embed
            n           = len(train_embed)
            nbatch      = nbatch if nbatch<n else n
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


def write_batch(batch, fname):
    # Write batch after predicting outputs
    # xxx relocate? xxx use string2undelim?
    batch_dump = pd.DataFrame({
        'stem':     [stem for stem in batch.stems],
        'output':   [output for output in batch.outputs],
        'morph':    [morph for morph in batch.morphs],
        'pred':     [pred for pred in batch.preds]
    })
    batch_dump.to_csv(fname, index=False)