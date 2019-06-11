#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Finley, S. (2015). Learning nonadjacent dependencies in phonology: Transparent vowels
# in vowel harmony. _Language_, 91(1), 48 - 72.

from tensormorphy.environ import config
import tensormorphy.phon_features as phon_features
from tensormorphy.symbol_embedder import SymbolEmbedder
from tensormorphy.role_embedder import RoleEmbedder
from tensormorphy.seq_embedder import SeqEmbedder
import pandas as pd
import sys

# Experiment 1 (Appendix A)
# N = 56 participants
# 5 repetitions of 24 training items
# Training: Stem/suffixed pairs
#   24 stems all with harmonic V1: 16 with harmonic V2, 8 with disharmonic /ε/
#   Opaque condition: V2 is /ε/ => suffix is /-e/
#   Transparent condition: V2 is /ε/ => suffix agrees in backness with V1
# Testing: 2AFC with old stems, new harmonic stems, new disharmonic stems
exp1 = {
    'consonants': ['p', 't', 'k', 'b', 'd', 'g', 'm', 'n'],
    'vowels': ['i', 'e', 'ε', 'o', 'u'],
    'front_harmonic': ['i', 'e'],
    'back_harmonic': ['u', 'o'],
    'disharmonic': ['ε'],
    'stem_shape': 'C1 V1 C2 V2 C3',
    'affixes': {'front':'-e', 'back':'-o'}
}

# Experiment 3 (Appendix B)
# N = 40 participants
# Like Experiment 1 except that disharmonic vowel is /ɪ/
exp3 = {
    'disharmonic': ['ɪ']
}

# Experiment 4 (Appendix B)
# N = 60 participants
# 4a: 10 repetitions of 24 training items
# 4b: +6 additional disharmonic items (= to total)
#   motɪp-motɪpo, dugɪb- dugɪbo, topɪm-topɪmo, nukɪt-nukɪto, konɪk-konɪko, bumɪg-bumɪgo
# 4c: each disharmonic item repeated twice

# Experiment 5 (Appendix B)
# N = 17 participants
# Like Experiment 4 except that disharmonic vowel is /ε/

# # # # #
# Model of experiment 4c
dat_dir = '/Users/colin/Dropbox/TensorProductStringToStringMapping/finley2015/'
dat_train = pd.read_csv(dat_dir +'exp34_training.csv', comment='#')
print (dat_train)