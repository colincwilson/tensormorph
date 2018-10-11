#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Container for role / unbinding, filler, morph embeddings
# and other global configurations.
# Provides batch binding and unbinding (i.e., query) operations on TPRs
# Convention: batch corresponds to *first* index in all inputs and outputs
# (see https://discuss.pytorch.org/t/how-to-repeat-a-vector-batch-wise)
# todo: convert ops to einsum (but currently does in-place operations!)


import torch
import torch.nn as nn
from torch import optim
from torch.nn import Parameter
from torch.nn.functional import hardtanh, linear, log_softmax, relu, relu6, logsigmoid, softmax, softplus
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch import tanh, sigmoid
from recorder import Recorder

import numpy as np
from numpy import linalg
import sys

import randVecs

def init(seq_embedder_, morph_embedder_):
    global seq_embedder, morph_embedder
    global F, R, U
    global random_fillers, random_roles
    global nfill, nrole, dfill, drole, dmorph
    global tau_min
    global loss_func
    global recorder
    global save_dir
    seq_embedder = seq_embedder_
    morph_embedder = morph_embedder_
    F, R, U = seq_embedder.F, seq_embedder.R, seq_embedder.U
    random_fillers = seq_embedder.random_fillers
    random_roles = seq_embedder.random_roles
    nfill, nrole = seq_embedder.nfill, seq_embedder.nrole
    dfill, drole = seq_embedder.dfill, seq_embedder.drole
    dmorph = morph_embedder.dmorph if morph_embedder is not None else 0
    tau_min = torch.zeros(1) + 0.0
    loss_func = ['loglik', 'euclid'][0]
    recorder = None


# batch dot (inner) product
def dot_batch(x, y):
    nbatch, nvec = x.shape
    val = torch.bmm(x.view(nbatch,1,nvec),\
                    y.view(nbatch,nvec,1))
    val = val.squeeze(1)
    return val
    #val = torch.einsum('aj,aj->a', [x, y])
    #val = val.unsqueeze(-1)
    #return val


# Gaussian radial basis function
# (note: does not include normalization constant)
# xxx move elsewhere
def rbf(x, mu, tau, log=False):
    s = -tau * torch.pow(x - mu, 2.0)
    if log: return s
    s = torch.exp(s)
    return s


# Botvinick-Watanabe (2007) scaled log normal function
# note: positions begin at 0 (cf. 1 in the original paper),
# therefore add 1 before taking logs ... wlog :)
# xxx move elsewhere
def scaledlognorm(x, mu, tau=torch.FloatTensor([1.0,]), log=False):
    s = torch.pow(torch.log(x + 1.0) - torch.log(mu + 1.0), 2.0)
    s = -tau * s    # equiv. to s / torch.pow(x,2.0)
    if log: return s
    s = torch.exp(s)
    return s


# map discrete position to column of matrix by look-up
# or soft position to column of matrix via attention
def posn2vec(M, posn, tau=None):
    if tau is None:
        return M[:,posn]
    attn = posn2attn(posn, tau, M.shape[-1])
    return attn2vec(attn)


# map discrete position to column of matrix by look-up
# or soft position to column of matrix via attention
# note: discrete position must be torch.LongTensor,
#       soft position must be torch.FloatTensor
# => output is nbatch x nrow(M)
def posn2vec_batch(M, posn, tau=None):
    if tau is None:
        return M[:,posn].t()
    attn = posn2attn_batch(posn, tau, M.shape[-1])
    val = attn2vec_batch(M, attn)
    return val
# example with discrete positions:
# posn = torch.LongTensor(np.array([0,3,5]))
# u = posn2vec_batch(U, posn)
# print u.data


# map attention distribution to column of matrix
def attn2vec(M, attn):
    val = torch.mm(M, attn)
    return val


# map attention distribution to column of matrix
def attn2vec_batch(M, attn):
    val = torch.mm(M, attn.t()).t()
    #val = torch.einsum('bj,ij->bi', [M, attn])
    return val


# map soft position to soft role vector
def posn2role(posn, tau=None):
    return posn2vec(R, posn, tau)


# map soft position to soft role vector
# => output is nbatch x drole
def posn2role_batch(posn, tau=None):
    return posn2vec_batch(R, posn, tau)


def attn2role_batch(attn):
    return attn2vec_batch(R, attn)


def attn2succ_batch(attn):
    return attn2vec_batch(S, attn)


# map soft position to soft unbinding vector
def posn2unbind(posn, tau=None):
    return posn2vec(U, posn, tau)


# map soft position to soft unbinding vector
# => output is nbatch x drole
def posn2unbind_batch(posn, tau=None):
    return posn2vec_batch(U, posn, tau)


# map attention distribution to soft unbinding vector
def attn2unbind_batch(attn):
    return attn2vec_batch(U, attn)


# unbind filler at hard or soft string position
def posn2filler(T, posn, tau=None):
    u = posn2unbind(posn, tau)
    f = T.mm(u)
    return f


# unbind filler at hard or soft string position
# for each position i in batch posn
#  - get unbinding vector ui by look-up or attention
#  - unbind filler fi from tpr Ti with ui
# => output is nbatch x nfill
def posn2filler_batch(T, posn, tau=None):
    u = posn2unbind_batch(posn, tau)
    #print T.shape, u.shape
    f = T.bmm(u.unsqueeze(2))
    #print f.shape
    f = f.squeeze(-1)
    #print f.shape
    return f


def attn2filler_batch(T, attn):
    u = attn2vec_batch(U, attn)
    f = T.bmm(u.unsqueeze(2))
    f = f.squeeze(-1)
    return f


# contruct filler-role bindings
# [https://discuss.pytorch.org/t/batch-outer-product/4025/2]
# => output is nbatch x nfill x nrole
def bind_batch(f, r):
    fr = torch.bmm(f.unsqueeze(2), r.unsqueeze(1))
    return fr


# normalize by summing over second dimension
# (NB. does not take absolute value of elements,
# therefore not equivalent to torch.normalize(1,1))
# xxx move elsewhere
def normalize(X, dim=1):
    Y = X / torch.sum(X, dim, keepdim=True)
    return Y


# normalize to length one
# xxx move elsewhere
def normalize_length(X, dim=1):
    Y = X / torch.mm(X.t(), X)
    return Y


# bound columns of each batch within [-1,1]
# xxx move elsewhere
def bound_batch_old(X):
        nbatch, m, n = X.shape
        Y = torch.zeros_like(X)
        ones = torch.ones((1, n))
        for i in xrange(nbatch):
            Xi = X[i,:,:]
            maxi = torch.max(Xi, 0)[0].view(1, n)
            mini = torch.min(Xi, 0)[0].view(1, n)
            maxi = torch.cat((maxi, -mini, ones), 0)
            maxi = torch.max(maxi, 0)[0].view(1, n)
            Y[i,:,:] = Xi / maxi
        return Y


# bound columns of each batch within [-1,1]
# xxx move elsewhere
def bound_batch(X):
    nbatch, m, n = X.shape
    ones = torch.ones((nbatch,1,n))
    maxi = torch.max(X, 1)[0].view(nbatch, 1, n)
    mini = torch.min(X,1)[0].view(nbatch, 1, n)
    maxi = torch.cat((maxi, -mini, ones), 1)
    maxi = torch.max(maxi, 1)[0].view(nbatch, 1, n)
    Y = X / maxi
    #delta = torch.sum(torch.sum(Y - rescale_batch(X), 0), 0)
    #print delta.data[0]
    return Y

# check that each value of vector is within bounds
# xxx move elsewhere
def check_bounds(x, min=0.0, max=1.0):
    if np.any(x<min) or np.any(x>max):
        return 0
    return 1
