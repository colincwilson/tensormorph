#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
import pandas as pd
import numpy as np
from collections import namedtuple
from unicodedata import normalize
import re, string, sys


FeatureMatrix = namedtuple('FeatureMatrix', 
    ['segments', 'vowels', 'features', 'ftr_matrix'])


def import_features(feature_matrix=None, segments=None, standardize=True):
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
        ftr_matrix = pd.read_csv(feature_matrix, encoding='utf-8')
    else:
        ftr_matrix = feature_matrix

    # List all segments and features in the matrix, find syllabic feature
    #ftr_matrix.iloc[:,0] = [normalize('NFC', x) for x in ftr_matrix.iloc[:,0]]
    segments_all = [x for x in ftr_matrix.iloc[:,0]]
    features_all = [x for x in ftr_matrix.columns[1:]]
    syll_ftr = [ftr for ftr in features_all if re.match('^(syl|syllabic)$', ftr)][0]

    # Normalize unicode insanity
    segments_all = [re.sub('\u0261', 'g', x) for x in segments_all] # script g -> g

    # Reduce feature matrix to given segments (if provided), pruning 
    # features other than vowel_ftr that have constant or redundant values
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

    # Syllabic segments
    vowels = [x for i,x in enumerate(segments) if 
        ftr_matrix[syll_ftr][i]=='+']

    # Convert features values to scalars
    for (key,val) in {'+':1.0, '-':-1.0, '0':0.0}.items():
        ftr_matrix = ftr_matrix.replace(to_replace=key, value = val)
    ftr_matrix = np.array(ftr_matrix.values)

    fm = FeatureMatrix(segments, vowels, features, ftr_matrix)
    if standardize:
        fm = standardize_matrix(fm)
    return fm


def make_one_hot_features(segments=None, vowels=None, standardize=True):
    """
    Create one-hot feature matrix from list of segments (or number of segments), 
    optionally standardizing with special symbols and features.
    """
    if isinstance(segments, int):
        segments = string.ascii_lowercase[:segments]
    features = segments[:]
    ftr_matrix = np.eye(len(segments))
    fm = FeatureMatrix(segments, vowels, features, ftr_matrix)
    if standardize:
        fm = standardize_matrix(fm)
    return fm


def standardize_matrix(fm):
    """
    Add special symbols (epsilon, begin, end) and 
    features (sym, begin, end, [C, V])
    """
    epsilon = config.epsilon
    stem_begin = config.stem_begin
    stem_end = config.stem_end
    segments = [epsilon, stem_begin, stem_end, *fm.segments]
    if fm.vowels is not None:
        vowels = fm.vowels
        consonants = [x for x in fm.segments if x not in vowels]
        features = ['sym', 'begin', 'end', 'C', 'V', *fm.features]
    else:
        vowels = None
        consonants = None
        features = ['sym', 'begin', 'end', *fm.features]

    ftr_matrix_ = np.zeros((len(segments), len(features)))
    ftr_matrix_[1:,0] = 1.0  # sym
    ftr_matrix_[1,1] = 1.0   # begin
    ftr_matrix_[2,2] = 1.0   # end

    if fm.vowels is not None:
        ftr_matrix_[3:,5:] = fm.ftr_matrix
        # (added 3 segments and 5 features)
        for i,x in enumerate(segments):
            ftr_matrix_[i,3] = 1.0 if x in consonants else 0.0
            ftr_matrix_[i,4] = 1.0 if x in vowels else 0.0
    else:
        ftr_matrix_[3:,3:] = fm.ftr_matrix
        # (added 3 segments and 3 features)
    
    fm = FeatureMatrix(segments, vowels, features, ftr_matrix_)
    return fm


def spec2vec(ftrspecs, feature_matrix=None):
    """
    Convert dictionary of feature specifications (ftr -> +/-/0) 
    to feature vector. If feature_matrix is omitted, defaults 
    to members of environ.config.
    """
    if feature_matrix is not None:
        features = feature_matrix.features
    else:
        features = config.ftrs
    
    specs = {'+':1.0, '-':-1.0, '0':0.0}
    n = len(features)
    vec = np.zeros(n)
    for ftr,spec in ftrspecs.items():
        i = features.index(ftr)
        vec[i] = specs[spec]
    return vec
    