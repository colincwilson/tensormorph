# -*- coding: utf-8 -*-

# Set up minimal environment for testing

import sys
sys.path.append('../src')
import environ
import phon_features
from form_embedder import FormEmbedder
from morphosyn_embedder import DummyEmbedder

feature_file = '../ftrs/' + ['hayes_features.csv', 'panphon_ipa_bases.csv'][0]
feature_matrix = phon_features.import_features(feature_file, ['p', 't', 'k', 's', 'm', 'n', 'r', 'i', 'e', 'a', 'o', 'u'])

symbol_params = {'feature_matrix': feature_matrix, }
role_params = {'nrole': 15, }
form_embedder = FormEmbedder(symbol_params, role_params)

morphosyn_embedder = DummyEmbedder()

environ.init(form_embedder, morphosyn_embedder)