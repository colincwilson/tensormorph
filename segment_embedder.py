#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
from .randvecs import randvecs
import torch
import pandas as pd
import sys
# xxx todo: rename nfill -> nsym, dfill -> dsym throughout


class SegmentEmbedder():
    """
    Create segment embeddings from a feature matrix (see phon_features), 
    optionally creating random embeddings
    Args:
        feature_matrix: as defined in phon_features
        randvec_params: arguments passed to randvecs
    """
    def __init__(self, feature_matrix=None, randvec_params=None):
        F = torch.tensor(feature_matrix.ftr_matrix.T, requires_grad=False, dtype=torch.float)
        dfill, nfill = F.shape
        if randvec_params is not None:
            randvec_params['n'] = nfill-3   # ignore epsilon, begin, end symbols
            randvec_params['dim'] = dfill-3 # and sym, begin, end features
            F1 = torch.FloatTensor(
                randvecs(**randvec_params)
            )
            F.data[3:,3:] = F1.data

        self.syms = feature_matrix.segments
        self.ftrs = feature_matrix.features
        self.ftr_matrix = feature_matrix.ftr_matrix
        self.F = F
        self.nfill = nfill
        self.dfill = dfill
        #print (F.shape, len(config.syms), len(config.ftrs))
 