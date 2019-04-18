#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Encoder-Decoder (after Bahdanau et al. 2014) implemented with OpenNMT-py
import torch
import torch.nn as nn
import onmt
from onmt.modules.embeddings import Embeddings
from collections import namedtuple
import string

# # # # # # # # # #
# Symbols, mapping from symbols to ids, and one-hot embeddings
stem_begin, stem_end = '⋊', '⋉'
syms = [stem_begin,] + [x for x in string.ascii_lowercase] + [stem_end,]
sym2id = {syms[i]:(i+1) for i in range(len(syms))}

def string2delim(x):
    y = [stem_begin,] + x.split(' ') + [stem_end,]
    return y

def string2idvec(x, delim=True):
    y = [sym2id[xi] for xi in x]
    y = torch.LongTensor(y)
    return y

def strings2idmat(xs):
    ys = [string2delim(x) for x in xs]
    ys.sort(key = lambda y : len(y), reverse=True)
    lens = torch.LongTensor([len(y) for y in ys])
    ids = [string2idvec(y) for y in ys]
    ids = nn.utils.rnn.pad_sequence(ids, batch_first=False, padding_value=0)
    ids = torch.LongTensor(ids) # (max_len, batch)
    ids = ids.unsqueeze(-1) # (max_len, batch, 1)
    return ids, lens

# Wrapper for embedding consistent with OpenNMT-py
class OneHotEmbeddings(Embeddings):
    def __init__(self, syms):
        super(Embeddings, self).__init__()
        F = torch.eye(len(syms)+1, requires_grad=False)
        F[0,0] = 0.0
        self.F = nn.Embedding.from_pretrained(F, freeze=True)
        self.syms = ['ε',] + syms
        self.nsym = len(self.syms)
        self.embedding_size = F.shape[0] # xxx check
    
    def forward(self, source, step=None):
        X = self.F(source.squeeze(-1))
        return X

embedding = OneHotEmbeddings(syms)
stems = ['l i g h t', 'd a r k']
stem_ids, stem_lens = strings2idmat(stems)
#Stems = embedding(stem_ids)
#print (Stems.shape) # (max_len, batch, embedding_size)
outputs = ['l i g h t n e s s', 'd a r k n e s s']
output_ids, output_lens = strings2idmat(outputs)
Outputs = embedding(output_ids)


# # # # # # # # # #
# Encoder-Decoder model
device = "cuda" if torch.cuda.is_available() else "cpu"
hidden_size = 50
learning_rate = 1

encoder = onmt.encoders.RNNEncoder(
    rnn_type='GRU',
    bidirectional=True,
    num_layers=1,
    hidden_size=hidden_size,
    dropout=0.0,
    embeddings=embedding)

#h, H, _ = encoder.forward(stem_ids, stem_lens)
#print (h)

decoder = onmt.decoders.StdRNNDecoder(
    rnn_type='GRU',
    bidirectional_encoder=True,
    num_layers=1,
    hidden_size=hidden_size,
    attn_type='mlp',
    attn_func='softmax',
    dropout=0.0,
    embeddings=embedding
)

model = onmt.models.NMTModel(
    encoder,
    decoder)
model.to(device)
#print (model)

#Pred,_ = model.forward(stem_ids, output_ids, stem_lens)
#print (Pred.shape)

model.generator = nn.Sequential(
    nn.Linear(hidden_size, embedding.nsym),
    nn.LogSoftmax(dim=-1)).to(device)

loss = onmt.utils.loss.NMTLossCompute(
    criterion=nn.NLLLoss(ignore_index=0, reduction="sum"),
    generator=model.generator)

optim = onmt.utils.optimizers.Optimizer(
    torch.optim.SGD(model.parameters(), lr=learning_rate),
    learning_rate=learning_rate,
    max_grad_norm=2)

report_manager = onmt.utils.ReportMgr(
    report_every=1,
    start_time=None,
    tensorboard_writer=None)

trainer = onmt.Trainer(
    model=model,
    train_loss=loss,
    valid_loss=loss,
    optim=optim,
    report_manager=report_manager)

Batch = namedtuple('Batch', ['src', 'tgt', 'batch_size'])
batch = Batch((stem_ids, stem_lens), output_ids, stem_lens.shape[0])
train_iter = [batch,]*100

print ('training ...')
trainer.train(
    train_iter=train_iter,
    train_steps=100,
    valid_iter=None,
    valid_steps=0)

Pred, _ = model.forward(stem_ids, output_ids, stem_lens)
Pred = model.generator(Pred) 
print (Pred.shape)
pred_max = torch.argmax(Pred, 2)
print (pred_max)
print (output_ids.squeeze(-1))