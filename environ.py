#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Initialize configs for role / unbinding / filler matrices,  
# morphosyntactic embeddings, learning parameters, etc.

import torch
import re, sys

# xxx use setattr(config, 'param_name', 'val') when reading  
# parameters from external file
class config:
    """
    Container for global settings
    """
    # Tensor-product representations
    epsilon             = u'ε'
    stem_begin          = u'⋊'
    stem_end            = u'⋉'
    random_fillers      = False
    random_roles        = False
    pass


def init(form_embedder=None, morphosyn_embedder=None):
    #from .seq_embedder   import SeqEmbedder
    from .morphosyn_embedder import MorphosynEmbedder
    from .decoder import Decoder, LocalistDecoder

    # Tensor-product representations
    config.form_embedder = form_embedder

    segment_embedder = form_embedder.segment_embedder
    for x in ['syms', 'ftrs', 'ftr_matrix', 'F', 'nfill', 'dfill']:
        setattr(config, x, getattr(segment_embedder, x))

    role_embedder = form_embedder.role_embedder
    for x in ['R', 'U', 'nrole', 'drole']:
        setattr(config, x, getattr(role_embedder, x))

    # Detect whether all feature values are in [0,1]
    # (i.e., whether all features are privative)
    ftrs_max = torch.max(config.F).item()
    ftrs_min = torch.min(config.F).item()
    config.privative_ftrs =\
        (0<=ftrs_min and ftrs_max<=1.0)
    #print ('privative features:', config.privative_ftrs); sys.exit(0)

    # Morphology embedding
    if morphosyn_embedder is None:
        config.morphosyn_embedder =\
            MorphosynEmbedder.get_embedder(None, None)        
    else:
        config.morphosyn_embedder = morphosyn_embedder
    config.dmorph = config.morphosyn_embedder.dmorph

    # Learning params
    config.nepoch       = 2000
    config.batch_size   = 64
    config.learn_rate   = 0.10
    config.dc           = 0.0
    config.lambda_reg   = 1.0e-5
    config.loss_func    = ['loglik', 'euclid'][0]

    # Miscellaneous params
    config.tau_min      = torch.zeros(1) + 0.0
    config.discretize   = False
    config.recorder     = None
    config.save_dir     = '/Users/colin/Desktop/tmorph_output'

    # Initialize decoder
    config.decoder      = Decoder() if 0 else LocalistDecoder()
