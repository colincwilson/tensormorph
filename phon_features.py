#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
import pandas as pd
import numpy as np
import re, sys

def read_features(feature_matrix=None, segments=None, standardize=True):
    """
    Process a feature matrix or feature file with segments in initial column. 
    If segments is specified, eliminates features that or constant of redundant. 
    If standardize flag is set, adds 'symbol existence (sym)' feature, features 
    to identify initial and final boundary symbols, and creates privative features 
    specifying consonants (C) and vowels (V); otherwise these features are assumed 
    to be already present.
    """

    # Read matrix from file if necessary
    if isinstance(feature_matrix, str):
        ftr_matrix = pd.read_csv(feature_matrix)
    else:
        ftr_matrix = feature_matrix

    # List all segments and features in the matrix, get syllabic feature
    segments_all = [x for x in ftr_matrix.iloc[:,0]]
    features_all = [x for x in ftr_matrix.columns[1:]]
    syll_ftr = [ftr for ftr in features_all if re.match('^(syl|syllabic)$', ftr)][0]

    # Reduce feature matrix to given segments (optional), pruning features 
    # other than vowel_ftr that have constant or redundant values
    if segments is not None:
        segments = [x for x in segments_all if x in segments]
        indx1 = [i for i,x in enumerate(segments_all) if x in segments]
        ftr_matrix = ftr_matrix.iloc[indx1,:]
        ftr_matrix.reset_index(drop=True)

        features_drop = []
        for j,ftr in enumerate(features_all):
            #print (ftr, ftr_matrix.loc[:,ftr].values)
            if ftr == syll_ftr:
                continue
            if ftr_matrix[ftr].nunique()<2:
                features_drop.append(ftr)
                continue
            for k in range(0, j):
                ftr2 = features_all[k]
                #print ('\t', features_all[k], ftr_matrix.loc[:,ftr2].values)
                if all(ftr_matrix.loc[:,ftr] == ftr_matrix.loc[:,ftr2]):
                # xxx also check for identity under negation (i.e., same partition)
                    features_drop.append(ftr)
                    break

        features = [ftr for ftr in features_all if ftr not in features_drop]
        ftr_matrix = ftr_matrix.loc[:,features]
        ftr_matrix = ftr_matrix.reset_index(drop=True)
    else:
        segments = segments_all
        features = features_all

    vowels = [x for i,x in enumerate(segments) if 
        ftr_matrix[syll_ftr][i]=='+'] # syllabic segments
    consonants = [x for x in segments if not x in vowels] # true consonants and glides

    # Convert features to numeric form (+1, -1, 0)
    for (key,val) in [('+', 1.0), ('-', -1.0), ('0', 0.0)]:
        ftr_matrix = ftr_matrix.replace(to_replace=key, value = val)
    ftr_matrix = np.array(ftr_matrix.values)

    if not standardize:
        return segments, features

    # Add special symbols (epsilon, begin, end) and 
    # features (sym, begin, end, C, V)
    # xxx split off
    epsilon = config.epsilon
    stem_begin = config.stem_begin
    stem_end = config.stem_end
    segments = [epsilon, stem_begin, *segments, stem_end] # xxx reorder
    features = ['sym', 'begin', 'end', 'C', 'V', *features]

    ftr_matrix_new = np.zeros((len(segments), len(features)))
    ftr_matrix_new[ 2:-1, 5: ] = ftr_matrix
    ftr_matrix = ftr_matrix_new

    ftr_matrix[1:,0] = 1.0  # sym
    ftr_matrix[1,1] = 1.0   # begin
    ftr_matrix[-1,2] = 1.0  # end
    for i,x in enumerate(segments):
        ftr_matrix[i,3] = 1.0 if x in consonants else 0.0
        ftr_matrix[i,4] = 1.0 if x in vowels else 0.0

    print ('segments:', segments)
    print ('features:', features)
    print ('feature matrix:\n', ftr_matrix.T)
    return segments, features, ftr_matrix.T