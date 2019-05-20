#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Initialize configs for role / unbinding / filler matrices,  
# morph embeddings, learning parameters, etc.

import torch
import re, sys

# container for global settings
# note: use setattr(config, 'param_name', 'val') when reading  
# parameters from external file
class config:
    epsilon = u'ε'
    pass

# xxx read params from external file
def init(features=None, data=None, morph_embedder=None, nrole=30, drole=30):
    from .seq_embedder   import SeqEmbedder
    from .morph_embedder import MorphEmbedder
    from .decoder        import Decoder, LocalistDecoder

    # tensor-product representations
    config.epsilon      = u'ε'
    config.stem_begin   = u'⋊'
    config.stem_end     = u'⋉'
    config.random_fillers = False
    config.random_roles = False
    config.nrole        = nrole
    config.drole        = drole
    if data is None:
        return

    seq_embedder        = SeqEmbedder(
        features = features,
        segments = data.segments,
        vowels = data.vowels,
        nrole = config.nrole
    )
    config.seq_embedder = seq_embedder

    # detect whether all feature values are in [0,1]
    # (i.e., whether all features are privative)
    ftrs_max = torch.max(config.F).item()
    ftrs_min = torch.min(config.F).item()
    config.privative_ftrs =\
        (0<=ftrs_min and ftrs_max<=1.0)
    #print ('privative features:', config.privative_ftrs); sys.exit(0)

    # morphology embedding
    if morph_embedder is None:
        config.morph_embedder =\
            MorphEmbedder.get_embedder(None, None)        
    else:
        config.morph_embedder = morph_embedder
    config.dmorph = config.morph_embedder.dmorph

    # learning
    config.nepoch       = 2000
    config.batch_size   = 64
    config.learn_rate   = 0.10
    config.dc           = 0.0
    config.lambda_reg   = 1.0e-5
    config.loss_func    = ['loglik', 'euclid'][0]

    # miscellaneous
    config.tau_min      = torch.zeros(1) + 0.0
    config.discretize   = False
    config.recorder     = None
    config.save_dir     = '/Users/colin/Desktop/tmorph_output'

    # prepare data
    config.data         = data
    data.split()
    data.embed()

    # initialize decoder
    config.decoder      = Decoder() if 0 else LocalistDecoder()
