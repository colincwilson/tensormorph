#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Encoder-Decoder (after Bahdanau et al. 2014) partly implemented with OpenNMT-py
import torch
import torch.nn as nn
#from torchcontrib.optim import SWA
import onmt
from onmt.modules.embeddings import Embeddings
import bahdanau_model
import sutskever_model
import pandas as pd
import numpy as np
from collections import namedtuple
import random, re, string, sys

model_type = ['bahdanau', 'sutskever'][1]

# # # # # # # # # #
# Symbols, mapping from symbols to ids, and one-hot embeddings
stem_begin, stem_end = '⋊', '⋉'
syms = ['ε', stem_begin,] + [x for x in string.ascii_lowercase] + [stem_end,]
sym2id = {syms[i]:i for i in range(len(syms))}

def string2delim(x):
    y = [stem_begin,] + x.split(' ') + [stem_end,]
    return y

def string2idvec(x, delim=True):
    y = [sym2id[xi] for xi in x]
    y = torch.LongTensor(y)
    return y

def strings2idmat(xs):
    ys = [string2delim(x) for x in xs]
    lens = torch.LongTensor([len(y) for y in ys])
    ids = [string2idvec(y) for y in ys]
    ids = nn.utils.rnn.pad_sequence(ids, batch_first=False, padding_value=0)
    ids = torch.LongTensor(ids) # (max_len, batch)
    ids = ids.unsqueeze(-1) # (max_len, batch, 1)
    return ids, lens

# Wrapper for embedding consistent with OpenNMT-py
# note: add 'non-epsilon' feature to break symmetry between nsym and embedding dimension
class OneHotEmbeddings(Embeddings):
    def __init__(self, syms):
        super(Embeddings, self).__init__()
        self.F = torch.cat([
            torch.eye(len(syms)),
            torch.ones(1,len(syms))], 0)
        # epsilon is all-zero vector
        self.F[0,0] = self.F[-1,0] = 0
        self.F = self.F.transpose(0,1)
        self.embedding = nn.Embedding(len(syms), len(syms)+1, padding_idx=0)
        self.embedding.weight = torch.nn.Parameter(self.F, requires_grad=False)
        self.syms = syms
        self.nsym = self.embedding.num_embeddings
        self.embedding_size = self.embedding.embedding_dim
        self.padding_idx = self.embedding.padding_idx
    
    def forward(self, source, step=None):
        source_embed = self.embedding(source.squeeze(-1))
        return source_embed

# # # # # # # # # #
# Data set, batches, batch generator
dat = pd.read_csv('/Users/colin/Dropbox/TensorProductStringToStringMapping/english/english_ness.csv')
dat['stem'] = [x.lower() for x in dat['stem']]
dat['stem_len'] = [len(x) for x in dat['stem']]
#dat['output_'] = [x+'ness' for x in dat['stem']] # stem+ness
dat['output'] = [x for x in dat['stem']] # stem
dat['output_'] = [x for x in dat['stem']]
dat = dat[(dat['output'] == dat['output_'])]
dat = dat[(dat['stem_len'] < 5)] # < 10
stems = [' '.join(x) for x in dat['stem']]
outputs = [' '.join(x) for x in dat['output']]

Batch = namedtuple('Batch', ['src', 'tgt', 'batch_size'])
def batchify(stems, outputs, batch_size=32):
    nbatch = int(np.ceil(len(stems) / batch_size))
    batch_indx = np.repeat([k for k in range(nbatch)], batch_size)
    batch_indx = random.sample([k for k in batch_indx], len(stems))
    batches = []
    for k in range(nbatch):
        examples = [(stems[i], outputs[i]) for i in range(len(stems))\
            if batch_indx[i] == k]
        examples.sort(key = lambda x : len(x[0]), reverse=True)
        stems_, outputs_ = zip(*examples)
        stem_ids, stem_lens = strings2idmat(stems_)
        output_ids, output_lens = strings2idmat(outputs_)
        batches.append(
            Batch((stem_ids, stem_lens), output_ids, stem_lens.shape[0])
        )
    return batches

# # # # # # # # # #
# Encoder-Decoder model
device = "cuda" if torch.cuda.is_available() else "cpu"
main_dir = '/Users/colin/Desktop/encoder_decoder_outputs/'
hidden_size = 150 # 100
train_model = True
learning_rate = 1.0
batch_size = 20
max_grad_norm = 2

embedding = OneHotEmbeddings(syms)
#print (embedding.F.shape)
#sys.exit(0)

if model_type=='bahdanau':
    model = bahdanau_model.BahdanauModel(
        embedding,
        hidden_size
    )
    # initalization (after Lee et al., 2015; Kann & Schutze, 2016)
    # xxx change to prior over weights instead of (only) initialization
    pseudo_identity_init = torch.cat([
            torch.eye(embedding.embedding_size),
            torch.zeros(hidden_size//2 - embedding.embedding_size, embedding.embedding_size)
    ], 0)
    pseudo_identity_init = torch.cat([
        pseudo_identity_init,
        pseudo_identity_init,
        pseudo_identity_init], 0)
    identity_init = torch.cat([
            torch.eye(hidden_size//2),
            torch.eye(hidden_size//2),
            torch.eye(hidden_size//2)], 0)
    zero_init = torch.zeros((hidden_size//2)*3)
    model.encoder.rnn.weight_ih_l0.data = pseudo_identity_init.clone().data
    model.encoder.rnn.weight_hh_l0.data = identity_init.clone().data
    model.encoder.rnn.weight_ih_l0_reverse.data = pseudo_identity_init.clone().data
    model.encoder.rnn.weight_hh_l0_reverse.data = identity_init.clone().data
    model.encoder.rnn.bias_ih_l0.data = zero_init.clone().data
    model.encoder.rnn.bias_hh_l0.data = zero_init.clone().data
    model.encoder.rnn.bias_ih_l0_reverse.data = zero_init.clone().data
    model.encoder.rnn.bias_hh_l0_reverse.data = zero_init.clone().data
else:
    model = sutskever_model.SutskeverModel(
        embedding,
        hidden_size
    )
#sys.exit(0)

loglik_loss = nn.NLLLoss(ignore_index=0, reduction="sum")

def batch_loss(gen_outputs, output_ids):
    output_len, batch_len, _ = gen_outputs.shape
    loss = loglik_loss(
        gen_outputs.view(output_len*batch_len, -1),
        output_ids[1:,:,:].view(-1)
    ) # note: collapse output posn and batch dimensions
    return loss

def batch2syms(batch_ids, add_stem_begin=False):
    nout, nbatch = batch_ids.shape
    xs = []
    for i in range(nbatch):
        xi = [syms[batch_ids[j,i]] for j in range(nout)]
        if add_stem_begin:
            xi = [stem_begin,] + xi
        xi = ' '.join(xi)
        xi = re.sub(stem_end+'.*', stem_end, xi)
        xs.append(xi)
    return xs

optim = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=1e-3)

def train(model, stems, outputs, batch_loss, optim):
    total_loss = 0.0
    batches = batchify(stems, outputs, batch_size)
    for batch in batches:
        stem_ids, stem_lengths = batch.src
        output_ids = batch.tgt
        gen_outputs,_ = model(stem_ids, output_ids, stem_lengths)
        loss = batch_loss(gen_outputs, output_ids)
        total_loss += loss.item()

        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optim.step()
    return total_loss

nepoch = 1 # 200
for epoch in range(nepoch):
    total_loss = train(model, stems, outputs, batch_loss, optim)
    print (total_loss)
    if (total_loss < 1.0):
        break

# testing
batches = batchify(stems, outputs, len(stems))
stem_ids, stem_lengths = batches[0].src
stem_embs = embedding(stem_ids.squeeze(-1))
output_ids = batches[0].tgt

if model_type=='bahdanau':
    _, stem_encs, _ = model.encoder(stem_ids)
    dec_outputs, dec_states, dec_attns =\
        model.decoder(stem_encs, output_ids, stem_lengths)
    gen_outputs, gen_pre_outputs =\
        model(stem_ids, output_ids, stem_lengths)
        #model.generator(dec_outputs)
    _, gen_ids = torch.max(gen_outputs, 2)
#print (np.round(gen_outputs[:,-1,:].data.numpy(), 3))
#print (torch.max(dec_attns[:,-1,:], 1))
else:
    model(stem_ids, output_ids, stem_lengths)
    torch.save(model.encoder.Z, main_dir+'sutskever_Zs.pt')
    for i in range(1,6):
        stem_lensi = torch.where(stem_lengths<i, stem_lengths, torch.ones_like(stem_lengths)*i)
        enc_finals, _, _ = model.encoder(stem_ids[0:i,:,:], stem_lensi)
        torch.save(enc_finals, main_dir+'sutskever_enc_finals'+str(i)+'.pt')
    gen_outputs, _ =\
        model(stem_ids, output_ids, stem_lengths)
    _, gen_ids = torch.max(gen_outputs, 2)

srcs = batch2syms(stem_ids.squeeze(-1))
targs = batch2syms(output_ids.squeeze(-1))
preds = batch2syms(gen_ids, add_stem_begin=True)
targ_preds = zip(srcs, targs, preds)
for x in targ_preds:
    print (x)

torch.save(embedding.F, main_dir+'F.pt')
torch.save(stem_ids, main_dir +'stem_ids.pt')
sys.exit(0)

torch.save(model.state_dict(), main_dir +'encoder_decoder_params.pt')
torch.save(stem_embs, main_dir +'stem_embs.pt')
torch.save(stem_encs, main_dir +'stem_encs.pt')
torch.save(dec_outputs, main_dir +'dec_outputs.pt')
torch.save(dec_states, main_dir +'dec_states.pt')
torch.save(dec_attns, main_dir +'dec_attns.pt')
torch.save(gen_pre_outputs, main_dir +'gen_pre_outputs.pt')
torch.save(gen_outputs, main_dir +'gen_outputs.pt')
torch.save(gen_ids, main_dir +'gen_ids.pt')