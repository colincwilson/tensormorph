#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Initializes configs for role / unbinding / filler matrices,  
# morph embeddings, learning parameters, etc.

import torch
import re, sys

# container for global settings
# note: use setattr(config, "param", "val") when reading  
# parameters from external file
class config: pass


# xxx read numerical params from external file
def init(data=None, features=None, morph_embedder=None):
    from seq_embedder   import SeqEmbedder
    from morph_embedder import MorphEmbedder
    from decoder        import Decoder, LocalistDecoder

    # tensor-product representations
    config.random_fillers = False
    config.random_roles = False
    config.nrole        = 30
    config.drole        = 30
    seq_embedder        = SeqEmbedder(
        segments=data.segments,
        vowels=data.vowels,
        nrole=config.nrole
    )
    config.seq_embedder = seq_embedder
    config.F            = seq_embedder.F
    config.R            = seq_embedder.R
    config.U            = seq_embedder.U
    config.nfill        = seq_embedder.nfill
    config.dfill        = seq_embedder.dfill

    # morphology embedding
    if morph_embedder is None:
        config.morph_embedder =\
            MorphEmbedder.get_embedder(None, None)        
    else:
        config.morph_embedder = morph_embedder
    config.dmorph = config.morph_embedder.dmorph

    # learning
    config.nepoch       = 1000
    config.batch_size   = 40
    config.learn_rate   = 0.1
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
