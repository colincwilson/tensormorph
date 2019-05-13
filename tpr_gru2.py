#!/usr/bin/env python
# -*- coding: utf-8 -*-

import environ, seq_embedder
from environ import config
import randVecs
import recurrent_layers
from decoder import Decoder
import torch
import torch.nn as nn
import numpy as np
import itertools, string, sys

n = 3   # number of ordinary fillers
k = 5   # number of roles
random_fillers, random_roles = 1, 0
main_dir = '/Users/colin/Desktop/tpr_gru_outputs/'

# Sequences
syms = [x for x in string.ascii_lowercase[0:n]]
dat = [ ]
for i in range(1,k-2+1):
    dat += list(itertools.product(*([syms]*i)))
dat = [' '.join(x) for x in dat]
print ('number of sequences: {}'.format(len(dat)))

# Sequence embeddings and tprs
environ.init()
seqembed = seq_embedder.SeqEmbedder(segments=syms, nrole=k, random_roles=random_roles)
if random_fillers:
    F_rand = torch.FloatTensor(
        randVecs.randVecs(config.nfill-1, config.nfill-1, np.eye(config.nfill-1))
    ) # random filler vectors
    config.F[1:,1:] = F_rand
#config.R = 1.0*config.R
#config.R = 1.0/float(k) * torch.tril(torch.ones(k,k))
#config.U = torch.FloatTensor(
#    np.linalg.inv(config.R.data.numpy().T)
#)
print ('max filler value: {}'.format(torch.max(torch.abs(config.F)).item()))
print ('max role value: {}'.format(torch.max(torch.abs(config.R)).item()))
tpr_max_value = k * torch.max(torch.abs(config.F)) * torch.max(torch.abs(config.R))
print ('maximum TPR value: {}'.format(tpr_max_value))
print (config.R.t() @ config.R)
#sys.exit(0)

dat_ = [seqembed.string2vec(x) for x in dat]
dat_embed = torch.stack([X.t() for X,_ in dat_], 1)
dat_len = torch.LongTensor([l for _,l in dat_])
dat_tpr = torch.stack([seqembed.string2tpr(x).flatten() for x in dat], 0)
print (dat_embed.shape)
print (dat_len.shape)
print (dat_tpr.shape)
print (dat_tpr[-1].reshape(config.dfill, config.drole))
#sys.exit(0)

# GRU model
model = recurrent_layers.GRU1(dat_embed.shape[-1], dat_tpr.shape[-1])
if 0:
    model.W_in.weight.data = torch.cat(
        [torch.eye(config.dfill)]*config.drole
    )
    #model.W_hz.weight.data = torch.cat([
    #    torch.zeros(config.drole*config.dfill, config.drole*(config.drole-1)),
    #    torch.FloatTensor(
    #        np.kron(np.eye(config.drole), np.ones((config.dfill,1)))
    #    )
    #], 1)
    print (model.W_hz.weight.shape)
    #gru = nn.GRU(dat_embed.shape[-1], dat_tpr.shape[-1])

loss_func = torch.nn.functional.mse_loss
optim = torch.optim.Adagrad(model.parameters(), lr=0.1)

config.tau_min = 0.0
decoder = Decoder(1.0)

# Train
for epoch in range(300):
    model.zero_grad()
    pred_tpr = model(dat_embed, dat_len)
    loss = loss_func(dat_tpr, pred_tpr)
    loss.backward()
    optim.step()
    if epoch % 10 == 0:
        print (loss.item(),)

# Evaluate
def printmat(X, digits=2):
    print (np.round(X.data.numpy(), digits))

def evaluate(dat_embed, dat_len, trim=True):
    pred_tpr = model(dat_embed, dat_len).reshape(-1, config.dfill, config.nrole)
    pred_idx = decoder.decode(pred_tpr)
    pred = [seqembed.idvec2string(x, trim=trim) for x in pred_idx]
    result = zip([seq_embedder.string2delim(x) for x in dat], pred)
    result = [x for x in result]
    return result

result = evaluate(dat_embed, dat_len)
errors = [x for x in result if x[0]!=x[1]]
accur = np.mean([x[0]==x[1] for x in result])
print ('errors:', errors)
print ('accuracy: {}'.format(accur))

torch.save(model.W_in.weight, main_dir+'W_in.pt')
torch.save(model.W_hz.weight, main_dir+'W_hz.pt')
torch.save(model.Z, main_dir+'Z.pt')
sys.exit(0)

Z = model.Z
printmat (Z[:,-1,:])
printmat (model.W_in.weight @ config.F[:,2] + model.W_in.bias)
printmat (model.W_in.weight, 0)
printmat (model.W_in.bias, 0)
printmat (model.W_hz.weight, 0)

result = evaluate(dat_embed, 4*torch.ones_like(dat_len), trim=False)
print (result)
sys.exit(0)

print (dat_tpr.shape, pred_tpr.shape, config.dfill, config.nrole, config.drole)
print ('dat:', dat)
print ('pred:', pred)

print (pred_tpr.shape)
dat_X = dat_tpr[-1,:].reshape(config.dfill, config.drole)
pred_X = pred_tpr[-1,:].reshape(config.dfill, config.drole)
print (dat_X.shape, pred_X.shape)
print (np.round(dat_X.data.numpy(),3))
print (np.round(pred_X.data.numpy(), 3))


pred_ = decoder.decode(pred_X.unsqueeze(0))
print (pred_)
print (seqembed.idvec2string(pred_))

#print (model.R[:,-1,:])
#print (np.round(model.Z[:,-1,:].data.numpy(),1))
#print (np.round(model.N[:,-1,:].data.numpy(),0))