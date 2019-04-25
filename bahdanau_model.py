#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Encoder-Decoder (after Bahdanau et al. 2014)
# see also https://bastings.github.io/annotated_encoder_decoder/
import torch
import torch.nn as nn
import onmt
from onmt.utils.misc import sequence_mask

class BahdanauDecoder(nn.Module):
    def __init__(self, embedding, nhidden):
        super(BahdanauDecoder, self).__init__()
        nsym = embedding.nsym
        self.embedding = embedding
        self.Ws = nn.Linear(nhidden, nhidden)           # state initialization
        self.Wz = nn.Linear(nsym+nhidden*2, nhidden)    # state update
        self.Wr = nn.Linear(nsym+nhidden*2, nhidden)    # state reset
        self.Wp = nn.Linear(nsym+nhidden*2, nhidden)    # state proposal
        self.Wa = nn.Linear(2*nhidden, nhidden)         # attention
        self.va = nn.Linear(nhidden, 1, bias=False)     #
        self.Wo = nn.Linear(nsym+nhidden*2, nhidden)    # output
        self.nsym = nsym
        self.nhidden = nhidden

    def forward(self, src_enc, tgt, src_lengths):
        dec_outputs = []
        dec_states = []
        dec_attns = []
        tgt_len = tgt.shape[0]
        src_mask = sequence_mask(src_lengths, max_len=src_enc.size(0)).transpose(0,1)

        # precompute all target-side embeddings
        tgt_embed = self.embedding(tgt.squeeze(-1))
        # initialize decoder state
        s = torch.tanh(self.Ws(src_enc[0,:,:]))
        #dec_states += [s.clone().unsqueeze(0),]
        # recurrence
        for i in range(1,tgt_len):
            # condition attention on current decoder state
            s_ = s.unsqueeze(0).expand(src_enc.shape[0],
                                        s.shape[0],
                                        s.shape[1])
            inpt = torch.cat([s_, src_enc], -1)
            ei = self.va(torch.tanh(self.Wa(inpt))).squeeze(-1)
            ei = ei.masked_fill(1 - src_mask, -float('inf'))
            ai = torch.exp(torch.log_softmax(ei, 0))
            #print (ai.shape, src_enc.shape); sys.exit(0)
            ci = torch.sum(ai.unsqueeze(-1) * src_enc, 0)

            # compute decoder output
            inpt = torch.cat([tgt_embed[i-1,:,:], s, ci], 1)
            ti = self.Wo(inpt)

            # store decoder state and output, attention distributions
            dec_states += [s.clone().unsqueeze(0),]
            dec_outputs += [ti.clone().unsqueeze(0),]
            dec_attns += [ai.clone().transpose(0,1).unsqueeze(0),]

            # update decoder state
            inpt = torch.cat([tgt_embed[i-1,:,:], s, ci], -1)
            zi = torch.sigmoid(self.Wz(inpt))   # update
            ri = torch.sigmoid(self.Wr(inpt))   # reset
            inpt = torch.cat([tgt_embed[i-1,:,:], ri * s, ci], -1)
            si = torch.tanh(self.Wp(inpt))      # proposal
            s = (1.0 - zi) * s  +  zi * si      # new state

        dec_outputs = torch.cat(dec_outputs)    # (tgt_len, batch_len, nhidden)
        dec_states = torch.cat(dec_states)      # (tgt_len, batch_len, nhidden)
        dec_attns = torch.cat(dec_attns)        # (tgt_len, batch_len, src_lengths)
        return dec_outputs, dec_states, dec_attns


class BahdanauModel(nn.Module):
    def __init__(self, embedding, nhidden):
        super(BahdanauModel, self).__init__()
        self.nsym = embedding.nsym
        self.nhidden = nhidden
        self.embedding = embedding
        self.encoder = onmt.encoders.RNNEncoder(
            rnn_type = 'GRU',
            bidirectional = True,
            num_layers = 1,
            hidden_size = self.nhidden,
            dropout = 0.0,
            embeddings = embedding)
        self.decoder = BahdanauDecoder(
            embedding,
            nhidden)
        self.generator = nn.Sequential(
            nn.Linear(nhidden, embedding.nsym),
            nn.LogSoftmax(dim=-1))

    def forward(self, src, tgt, src_lengths, bptt=False):
        # encode source
        _, src_enc, _ = self.encoder(src)
        # decode
        dec_outputs, dec_states, dec_attns =\
            self.decoder(src_enc, tgt, src_lengths)
        # generate from decoder outputs
        gen_outputs = self.generator(dec_outputs)
        
        return gen_outputs
