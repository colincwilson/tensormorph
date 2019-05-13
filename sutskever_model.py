#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Encoder-Decoder (after Sutskever et al. 2014)
# initial state of decoder is final state of encoder
import torch
import torch.nn as nn
import onmt
from onmt.utils.misc import sequence_mask
from recurrent_layers import GRU1
from bahdanau_model import BahdanauGenerator

class SutskeverModel(nn.Module):
    def __init__(self, embedding, nhidden):
        super(SutskeverModel, self).__init__()
        self.nsym = embedding.nsym
        self.nemb = embedding.embedding_size
        self.nhidden = nhidden
        self.embedding = embedding
        if 0:
            self.encoder = onmt.encoders.RNNEncoder(
                rnn_type = 'GRU',
                bidirectional = False,
                num_layers = 1,
                hidden_size = self.nhidden,
                dropout = 0.0,
                embeddings = embedding)
        else:
            self.encoder = GRU1(
                input_size = self.nemb,
                hidden_size = nhidden
            )
        self.decoder = torch.nn.GRU(
            input_size = 0,
            hidden_size = nhidden,
            num_layers = 1,
            bias = True,
            batch_first = False,
            dropout = 0.0,
            bidirectional = False
        )
        self.generator = BahdanauGenerator(
            embedding,
            nhidden)

    def forward(self, src, tgt, src_lengths, bptt=False):
        # encode source
        src_embed = self.embedding(src)
        enc_final = self.encoder(src_embed, src_lengths)
        if enc_final.shape[0]>1:
            enc_final = torch.cat(torch.split(0),-1).unsqueeze(0)
        # decode
        dec_outputs, _ =\
            self.decoder(torch.zeros(tgt.shape[0], tgt.shape[1], 0), enc_final)
        # generate from decoder outputs
        gen_outputs, gen_pre_outputs =\
            self.generator(dec_outputs)
        gen_outputs, gen_pre_outputs =\
            gen_outputs[1:,:,:], gen_pre_outputs[1:,:,:]
        return gen_outputs, gen_pre_outputs