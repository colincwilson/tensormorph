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
        nemb = embedding.embedding_size
        self.embedding = embedding
        # todo: clean up
        self.Ws = nn.Linear(nhidden, nhidden)           # state initialization
        self.Wz = nn.Linear(nemb+nhidden*2, nhidden)    # state update
        self.Wr = nn.Linear(nemb+nhidden*2, nhidden)    # state reset
        self.Wn = nn.Linear(nemb+nhidden*2, nhidden)    # state proposal
        self.Wa = nn.Linear(2*nhidden, nhidden)         # attention
        self.va = nn.Linear(nhidden, 1, bias=False)     #
        self.Wo = nn.Linear(nemb+nhidden*2, nhidden)    # output
        self.nsym = nsym
        self.nemb = nemb
        self.nhidden = nhidden

    def forward(self, src_enc, tgt, src_lengths):
        dec_outputs = []
        dec_states = []
        dec_attns = []
        tgt_len = tgt.shape[0]
        src_mask = sequence_mask(src_lengths, max_len=src_enc.size(0)).transpose(0,1)

        # Precompute all target-side embeddings
        tgt_embed = self.embedding(tgt.squeeze(-1))
        # Initialize decoder state
        s = torch.tanh(self.Ws(src_enc[0,:,:]))
        #dec_states += [s.clone().unsqueeze(0),]
        # Recurrence
        for i in range(1,tgt_len):
            # Condition attention on current decoder state
            s_ = s.unsqueeze(0).expand(src_enc.shape[0],
                                        s.shape[0],
                                        s.shape[1])
            inpt = torch.cat([s_, src_enc], -1)
            ei = self.va(torch.tanh(self.Wa(inpt))).squeeze(-1)
            ei = ei.masked_fill(1 - src_mask, -float('inf'))
            ai = torch.exp(torch.log_softmax(ei, 0))
            #print (ai.shape, src_enc.shape); sys.exit(0)
            ci = torch.sum(ai.unsqueeze(-1) * src_enc, 0)

            # Compute decoder output (single layer, no drop-out)
            # note: use encoder state s_(i-1), before state update
            inpt = torch.cat([tgt_embed[i-1,:,:], s, ci], 1)
            ti = self.Wo(inpt)

            # Store decoder state and output, attention distributions
            dec_states += [s.clone().unsqueeze(0),]
            dec_outputs += [ti.clone().unsqueeze(0),]
            dec_attns += [ai.clone().transpose(0,1).unsqueeze(0),]

            # Update decoder state
            inpt = torch.cat([tgt_embed[i-1,:,:], s, ci], -1)
            zi = torch.sigmoid(self.Wz(inpt))   # update
            ri = torch.sigmoid(self.Wr(inpt))   # reset
            inpt = torch.cat([tgt_embed[i-1,:,:], ri * s, ci], -1)
            ni = torch.tanh(self.Wn(inpt))      # proposal
            s = (1.0 - zi) * s  +  zi * ni      # new state

        dec_outputs = torch.cat(dec_outputs)    # (tgt_len, batch_len, nhidden)
        dec_states = torch.cat(dec_states)      # (tgt_len, batch_len, nhidden)
        dec_attns = torch.cat(dec_attns)        # (tgt_len, batch_len, src_lengths)
        return dec_outputs, dec_states, dec_attns


class BahdanauGenerator(nn.Module):
    def __init__(self, embedding, nhidden):
        super(BahdanauGenerator, self).__init__()
        self.embedding = embedding
        self.nhidden = nhidden
        self.nemb = embedding.embedding_dim
        self.Wg = nn.Linear(nhidden, self.nemb, bias=True)
        # alt: use Bilinear directly
    
    def forward(self, dec_outputs):
        # Bilinear map comparing dec_outputs and output embeddings
        gen_pre_outputs = self.Wg(dec_outputs)
        gen_outputs = torch.matmul(gen_pre_outputs, self.embedding.weight.t())
        #print (gen_pre_outputs.shape, self.embedding.F.transpose(0,1).shape); sys.exit(0)
        # Convert to probability distribution over output symbols
        gen_outputs = torch.log_softmax(gen_outputs, dim = -1)
        return gen_outputs, gen_pre_outputs


class BahdanauModel(nn.Module):
    def __init__(self, embedding, nhidden):
        super(BahdanauModel, self).__init__()
        self.nsym = embedding.nsym
        self.nemb = embedding.embedding_dim
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
        self.generator = BahdanauGenerator(
            embedding,
            nhidden)

    def forward(self, src, tgt, src_lengths, bptt=False):
        # Encode source
        _, src_enc, _ = self.encoder(src)
        # Decode
        dec_outputs, dec_states, dec_attns =\
            self.decoder(src_enc, tgt, src_lengths)
        # Generate from decoder outputs
        gen_outputs, gen_pre_outputs =\
            self.generator(dec_outputs)
        return gen_outputs, gen_pre_outputs
