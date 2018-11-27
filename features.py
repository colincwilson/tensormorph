#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
import torch

class FeatureMatrix():
    def __init__(self, ftr_matrix=None, segments=None, vowels=None):
        if ftr_matrix is not None:
            parse_feature_matrix(ftr_matrix)

def parse_feature_matrix(ftr_matrix):
    segments    = [x for x in ftr_matrix.segment]
    features    = [x for x in ftr_matrix.columns[1:]]
    epsilon     = config.epsilon
    stem_begin  = config.stem_begin
    stem_end    = config.stem_end

    syms = [epsilon, stem_begin] + segments + [stem_end,]
    print (syms)
    print (features)