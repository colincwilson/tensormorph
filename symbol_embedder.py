#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
from .randvecs import randvecs
import torch
import pandas as pd
import re, sys

# xxx todo: change nfill -> nsym, dfill -> dsym throughout

class SymbolEmbedder():
    """
    Create segment embeddings from a preloaded feature matrix or feature file, 
    otherwise create one-hot or random embeddings from a list of segments.
    Adds a 'symbol existence (sym)' feature, features to identify initial and 
    final boundary symbols, and either detects or can create privative features 
    specifying consonants (C) and vowels (V); the existence and delimiter features 
    and special symbols (epsilon, delimiters) are purely internal should not be 
    included in input feature matrices or files.
    Args (all optional):
        feature_matrix: pandas DataFrame or filename
        segments (list): ordinary symbols
        vowels (list): ordinary symbols that are vowels
        randvec_params (list): arguments passed to randvecs
    """
    def __init__(self, feature_matrix=None, segments=None, 
                        vowels=None, randvec_params=None):
        if feature_matrix is not None:
            if isinstance(feature_matrix, str):
                feature_matrix = pd.read_csv(feature_matrix)
            (syms, ftrs, nfill, dfill, F) =\
                self.parse_feature_matrix(feature_matrix)
        else:
            (syms, ftrs, nfill, dfill, F) =\
                self.make_embedding(segments, vowels, randvec_params)
        config.syms     = syms
        config.ftrs     = ftrs
        config.nfill    = nfill
        config.dfill    = dfill
        config.F        = F


    def parse_feature_matrix(self, ftr_matrix):
        """
        Read features from a pandas DataFrame with initial column named 'segment', 
        detecting the feature that distinguishes vowels from consonants.
        """
        epsilon     = config.epsilon
        stem_begin  = config.stem_begin
        stem_end    = config.stem_end
        segments    = [x for x in ftr_matrix.segment]
        features    = [x for x in ftr_matrix.columns[1:]]
        vowel_ftr   = [ftr for ftr in features if re.match('^(syll|syllabic)$', ftr)][0]
        vowels      = [x for i,x in enumerate(segments) if ftr_matrix[vowel_ftr][i]=='+']
        consonants  = [x for x in segments if x not in vowels] # true consonants and glides

        syms    = [epsilon, stem_begin] + segments + [stem_end,]
        ftrs    = ['sym', 'begin', 'end', 'C', 'V'] + features
        nfill   = len(syms)     # segments + special symbols
        dfill   = len(ftrs)     # ordinary + special features

        F = torch.zeros((dfill, nfill))
        F.data[0,1:]    = 1.0   # non-epsilon feature
        F.data[1,1]     = 1.0   # stem-begin feature
        F.data[2,-1]    = 1.0   # stem-end feature
        for j,sym in enumerate(syms):   # C and V features
            F.data[3,j] = 1.0 if sym in consonants else 0.0
            F.data[4,j] = 1.0 if sym in vowels else 0.0
        for ftr in features:
            i = ftrs.index(ftr)
            for j,val in enumerate(ftr_matrix[ftr]):
                # todo: skip vowel_ftr
                # xxx enforce privativity?
                if val == '+':
                    F.data[i,j+2] = +1.0
                if val == '-':
                    F.data[i,j+2] = -1.0
        
        return (syms, ftrs, nfill, dfill, F)


    def make_embedding(self, segments=None, vowels=None, randvec_params=None):
        """
        Construct symbol embedding from list of segments, etc.
        """
        epsilon = config.epsilon
        stem_begin = config.stem_begin
        stem_end = config.stem_end
        
        # Collect ordinary and special symbols and features
        syms = [epsilon, stem_begin] + segments + [stem_end,]
        ftrs = ['sym', 'begin']
        if vowels is not None:
            ftrs += ['C', 'V']
        ftrs += segments + ['end',]
        nfill = len(syms)
        dfill = len(ftrs)

        # Embedding matrix
        F = torch.zeros(dfill, nfill, requires_grad=False)
        F.data[0,1:] = 1.0   # Symbol existence feature
        F.data[1,1] = 1.0    # Stem-begin feature
        F.data[-1,-1] = 1.0  # Stem-end feature

        # C and V features
        if vowels is not None:
            for x in segments:
                j = syms.index(x)
                F.data[2,j] = 1.0 if (x not in vowels) else 0.0
                F.data[3,j] = 1.0 if (x in vowels) else 0.0

        # Random or one-hot embeddings of ordinary symbols
        if randvec_params is None:
            F1 = torch.eye(nfill-3)
        else:
            randvec_params['n'] = nfill-3
            randvec_params['dim'] = nfill-3
            F1 = torch.FloatTensor(
                randvecs(**randvec_params)
            )

        i = 2 if vowels is None else 4
        j = 2
        F.data[i:-1,j:-1] = F1.data

        return (syms, ftrs, nfill, dfill, F)
