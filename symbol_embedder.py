#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
import torch
import re, sys

class SymbolEmbedder():
    def __init__(self, ftr_matrix=None, segments=None, vowels=None):
        if ftr_matrix is not None:
            (syms, ftrs, nfill, dfill, F) =\
                self.parse_feature_matrix(ftr_matrix)
        else:
            (syms, ftrs, nfill, dfill, F) =\
                self.get_atomic_embedding(segments, vowels)
        config.syms     = syms
        config.ftrs     = ftrs
        config.nfill    = nfill
        config.dfill    = dfill
        config.F        = F


    def parse_feature_matrix(self, ftr_matrix):
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
        dfill   = len(ftrs)     # regular + special features

        F = torch.zeros((dfill, nfill))
        F.data[0,1:]    = 1.0   # non-epsilon feature
        F.data[1,1]     = 1.0   # stem-begin feature
        F.data[2,-1]    = 1.0   # stem-end feature
        for j,sym in enumerate(syms):
            F.data[3,j] = 1.0 if sym in consonants else 0.0
            F.data[4,j] = 1.0 if sym in vowels else 0.0
        for ftr in features:
            i = ftrs.index(ftr)
            for j,val in enumerate(ftr_matrix[ftr]):
                # xxx enforce privativity?
                if val == '+':
                    F.data[i,j+2] = +1.0
                if val == '-':
                    F.data[i,j+2] = -1.0
        
        return (syms, ftrs, nfill, dfill, F)


    def get_atomic_embedding(self, segments, vowels):
        epsilon     = config.epsilon
        stem_begin  = config.stem_begin
        stem_end    = config.stem_end
        consonants = [x for x in segments if x not in vowels]

        syms    = [epsilon, stem_begin] + segments + [stem_end,]
        ftrs    = ['sym', 'begin', 'end', 'C', 'V'] + segments
        nfill   = len(syms)
        dfill   = len(ftrs)

        F = torch.zeros((dfill, nfill))
        F.data[0,1:]    = 1.0   # non-epsilon feature
        F.data[1,1]     = 1.0   # stem-begin feature
        F.data[2,-1]    = 1.0   # stem-end feature
        for j,sym in enumerate(syms):
            F.data[3,j] = 1.0 if sym in consonants else 0.0
            F.data[4,j] = 1.0 if sym in vowels else 0.0
        for j in range(len(segments)):
            F.data[j+5, j+2] = 1.0
        
        return (syms, ftrs, nfill, dfill, F)