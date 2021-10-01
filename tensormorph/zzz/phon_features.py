# -*- coding: utf-8 -*-

import re, string, sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import namedtuple
from unicodedata import normalize
import config
# todo: delegate to panphon or phoible if possible
# todo: warn about missing/nan feature values in matrix

FeatureMatrix = namedtuple('FeatureMatrix',
                           ['segments', 'vowels', 'features', 'ftr_matrix'])


def import_features(feature_matrix=None, segments=None, standardize=True):
    """
    Process a feature matrix or feature file with segments in initial column. 
    If segments is specified, eliminates constant and redundant features. 
    If standardize flag is set:
    - Add epsilon symbol with all-zero feature vector,
    - Add symbol-presence feature (sym),
    - Add begin/end delimiters and feature to identify them (begin:+1, end:-1),
    - Add feature to identify consonants (C) and vowels (V) (C:+1, V:-1),
    - Add wildcard symbol with mean of other feature vectors [optional]
    Otherwise these symbols and features are assumed to be already present 
    in the feature matrix or file.
    todo: arrange segments in IPA order
    """

    # Read matrix from file or use arg matrix
    if isinstance(feature_matrix, str):
        ffeature_matrix = Path(config.feature_dir) / feature_matrix
        #ffeature_matrix = moduledir/('../ftrs/'+ feature_matrix)
        ftr_matrix = pd.read_csv(ffeature_matrix, encoding='utf-8')
    else:
        ftr_matrix = feature_matrix

    # Add long segments and length feature ("let there be colons")
    ftr_matrix_short = ftr_matrix.copy()
    ftr_matrix_long = ftr_matrix.copy()
    ftr_matrix_short['long'] = '-'
    ftr_matrix_long['long'] = '+'
    ftr_matrix_long.iloc[:, 0] = [x + 'ː' for x in ftr_matrix_long.iloc[:, 0]]
    ftr_matrix = pd.concat([ftr_matrix_short, ftr_matrix_long],
                           axis=0,
                           sort=False)

    # List all segments and features in the matrix, find syllabic feature,
    # and remove first column (containing segments)
    # ftr_matrix.iloc[:,0] = [normalize('NFC', x) for x in ftr_matrix.iloc[:,0]]
    segments_all = [x for x in ftr_matrix.iloc[:, 0]]
    features_all = [x for x in ftr_matrix.columns[1:]]
    syll_ftr = [
        ftr for ftr in features_all if re.match('^(syl|syllabic)$', ftr)
    ][0]
    ftr_matrix = ftr_matrix.iloc[:, 1:]

    # Normalize unicode [partial]
    # no script g, no tiebars, ...
    ipa_substitutions = {'\u0261': 'g', 'ɡ': 'g', '͡': ''}
    for (s, r) in ipa_substitutions.items():
        segments_all = [re.sub(s, r, x) for x in segments_all]
    #print('segments_all:', segments_all)

    # Handle segments with diacritics [partial]
    diacritics = [
        ("ʼ", ('constr.gl', '+')),
        ("ʰ", ('spread.gl', '+')),
        ("[*]", ('constr.gl', '+')),  # Korean
        ("ʷ", ('round', '+'))
    ]
    diacritic_segs = []
    if segments is not None:
        for seg in segments:
            # Detect and strip diacritics
            base_seg = seg
            diacritic_ftrs = []  # features marked by diacritics
            for (diacritic, ftrval) in diacritics:
                if re.search(diacritic, base_seg):
                    diacritic_ftrs.append(ftrval)
                    base_seg = re.sub(diacritic, '', base_seg)
            if len(diacritic_ftrs) == 0:
                continue
            # Specify diacritic features
            idx = segments_all.index(base_seg)
            base_ftr = [x for x in ftr_matrix.iloc[idx, :]]
            for ftr, val in diacritic_ftrs:
                idx = features_all.index(ftr)
                base_ftr[idx] = val
            diacritic_segs.append((seg, base_ftr))
        # Add segments with diacritics and features
        if len(diacritic_segs) > 0:
            new_segs = [x[0] for x in diacritic_segs]
            new_ftr_vecs = pd.DataFrame([x[1] for x in diacritic_segs])
            new_ftr_vecs.columns = ftr_matrix.columns
            segments_all += new_segs
            ftr_matrix = pd.concat([ftr_matrix, new_ftr_vecs],
                                   ignore_index=True)
        #print(segments_all)
        #print(ftr_matrix)

    # Reduce feature matrix to observed segments (if provided), pruning
    # features other than syll_ftr that have constant or redundant values
    if segments is not None:
        # Check that all segments appear in the feature matrix
        missing_segments = [x for x in segments if x not in segments_all]
        if len(missing_segments) > 0:
            print(f'Segments missing from feature matrix: '
                  f'{missing_segments}')

        segments = [x for x in segments_all if x in segments]
        ftr_matrix = ftr_matrix.loc[[x in segments for x in segments_all], :]
        ftr_matrix.reset_index(drop=True)

        features = [ftr for ftr in ftr_matrix.columns \
            if ftr == 'syll_ftr' or ftr_matrix[ftr].nunique() > 1]
        ftr_matrix = ftr_matrix.loc[:, features]
        ftr_matrix = ftr_matrix.reset_index(drop=True)
    else:
        segments = segments_all
        features = features_all

    # Syllabic segments
    vowels = [ x for i, x in enumerate(segments) \
        if ftr_matrix[syll_ftr][i] == '+']

    # Convert feature values to scalars
    ftr_specs = {'+': 1.0, '-': -1.0, '0': 0.0}
    for (key, val) in ftr_specs.items():
        ftr_matrix = ftr_matrix.replace(to_replace=key, value=val)
    ftr_matrix = np.array(ftr_matrix.values)

    fm = FeatureMatrix(segments, vowels, features, ftr_matrix)
    if standardize:
        fm = standardize_matrix(fm)

    # Write feature matrix to data folder
    ftr_matrix_out = pd.DataFrame(
        np.round(fm.ftr_matrix), index=fm.segments, columns=fm.features)
    for (key, val) in ftr_specs.items():
        ftr_matrix_out = ftr_matrix_out.replace(to_replace=val, value=key)
    ftr_matrix_out.to_csv(config.fdata.with_suffix('.ftr'), sep='\t')

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

    # todo: Write feature matrix to data folder

    return fm


def standardize_matrix(fm):
    """
    Add special symbols (epsilon, bos, eos) and features (sym, begin/end, C/V)
    """
    epsilon, bos, eos, wildcard = \
        config.epsilon, config.bos, config.eos, config.wildcard
    segments = [epsilon, bos, eos, *fm.segments]
    if fm.vowels is not None:
        vowels = fm.vowels
        consonants = [x for x in fm.segments if x not in vowels]
        features = ['sym', 'begin/end', 'C/V', *fm.features]
        config.sym_ftr = sym_ftr = 0
        config.edge_ftr = delim_ftr = 1
        config.cv_ftr = cv_ftr = 2
    else:  # todo: require vowels
        vowels = None
        consonants = None
        features = ['sym', 'begin/end', *fm.features]
        config.sym_ftr, config.edge_ftr = 0, 1

    ftr_matrix_ = np.zeros((len(segments), len(features)))
    ftr_matrix_[1:, sym_ftr] = +1.0  # sym
    ftr_matrix_[1, delim_ftr] = +1.0  # begin
    ftr_matrix_[2, delim_ftr] = -1.0  # end

    if fm.vowels is not None:
        # (add 3 segments and 3 features)
        ftr_matrix_[3:, 3:] = fm.ftr_matrix
        for i, x in enumerate(segments):
            if x in consonants:
                ftr_matrix_[i, cv_ftr] = +1.0
            elif x in vowels:
                ftr_matrix_[i, cv_ftr] = -1.0
    else:  # todo: require vowels
        # (add 3 segments and 2 features)
        ftr_matrix_[3:, 2:] = fm.ftr_matrix

    # Wildcard symbol
    if 0:
        segments.append(wildcard)
        ftr_matrix__ = np.zeros((len(segments), len(features)))
        ftr_matrix__[-1, :] = np.mean(ftr_matrix_[1:-1, :], 0)
        ftr_matrix__[:-1, :] = ftr_matrix_
    else:
        ftr_matrix__ = ftr_matrix_

    fm = FeatureMatrix(segments, vowels, features, ftr_matrix__)
    return fm


# xxx not used?
def ftrspec2vec(ftrspecs, feature_matrix=None):
    """
    Convert dictionary of feature specifications (ftr -> +/-/0) 
    to feature + 'attention' vectors.
    If feature_matrix is omitted, default to environ.config.
    """
    if feature_matrix is not None:
        features = feature_matrix.features
    else:
        features = config.ftrs

    specs = {'+': 1.0, '-': -1.0, '0': 0.0}
    n = len(features)
    w = np.zeros(n)
    a = np.zeros(n)
    for ftr, spec in ftrspecs.items():
        if spec == '0':
            continue
        i = features.index(ftr)
        if i < 0:
            print('ftrspec2vec: could not find feature', ftr)
        w[i] = specs[spec]  # non-zero feature specification
        a[i] = 1.0  # 'attention' weight identifying non-zero feature
    return w, a
