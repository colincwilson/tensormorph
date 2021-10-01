# -*- coding: utf-8 -*-

import config
from randVecs import randvecs
import torch
import pandas as pd
import sys


class SymbolEmbedder():
    """
    Create symbol embeddings from a feature matrix (see phon_features), 
    or random vectors
    Args:
        feature_matrix: as defined in phon_features
        randvec_params: arguments passed to randvecs
    """

    def __init__(self, feature_matrix=None, randvec_params=None):
        F = torch.tensor(
            feature_matrix.ftr_matrix_vec.T,  # transpose to segments in columns
            requires_grad=False,
            dtype=torch.float,
            device=config.device)
        dsym, nsym = F.shape
        if randvec_params is not None:
            randvec_params['n'] = nsym - 3  # ignore epsilon, bos, eos
            randvec_params['dim'] = dsym - 3  # and sym, begin/end, C/V features
            F1 = torch.FloatTensor(
                randvecs(**randvec_params), device=config.device)
            F.data[3:, 3:] = F1.data

        self.syms = feature_matrix.symbols
        self.ftrs = feature_matrix.features
        self.ftr_matrix = feature_matrix.ftr_matrix
        self.ftr_matrix_vec = feature_matrix.ftr_matrix_vec
        self.F = F
        self.nsym = nsym
        self.dsym = dsym

        print(f'symbols {feature_matrix.symbols} '
              f'({len(feature_matrix.symbols)})')
        print(f'vowels {feature_matrix.vowels} '
              f'({len(feature_matrix.vowels)})')
        print(f'features {feature_matrix.features} '
              f'({len(feature_matrix.features)})')
        print(f'feature matrix\n' f'{feature_matrix.ftr_matrix_vec}')

        #print(F.shape, len(self.ftrs), len(self.syms)); sys.exit(0)