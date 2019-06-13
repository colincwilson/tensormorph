#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provides batch binding and unbinding (i.e., query) operations on TPRs
Convention: batch corresponds to *first* index in all inputs and outputs
(see https://discuss.pytorch.org/t/how-to-repeat-a-vector-batch-wise)
todo: convert ops to einsum (but currently does in-place operations?)
"""

import torch
import torch.nn as nn
from torch import optim
from torch.nn import Parameter
from torch.nn.functional import hardtanh, linear, log_softmax, relu, relu6, logsigmoid, softmax, softplus
from torch.distributions import RelaxedBernoulli
from torch import tanh, sigmoid

from .environ import config
from .recorder import Recorder

import numpy as np
from numpy import linalg
import sys

def dot_batch(x, y):
    """
    Batch dot (inner) product.
    """
    batch_size, dim = x.shape
    val = torch.bmm(x.view(batch_size, 1, dim),\
                    y.view(batch_size, dim, 1))
    val = val.squeeze(1)
    return val
    #val = torch.einsum('aj,aj->a', [x, y])
    #val = val.unsqueeze(-1)
    #return val


def rbf(x, mu, tau, log=False):
    """
    Gaussian radial basis function
    (note: does not include normalization constant).
    todo: relocate
    """
    s = -tau * torch.pow(x - mu, 2.0)
    if log: return s
    s = torch.exp(s)
    return s


def scaledlognorm(x, mu, tau=torch.FloatTensor([1.0,]), log=False):
    """
    Botvinick-Watanabe (2007) scaled log normal function
    note: positions begin at 0 (cf. 1 in the original paper),
    therefore add 1 before taking logs ... wlog :)
    todo: relocate
    """
    s = torch.pow(torch.log(x + 1.0) - torch.log(mu + 1.0), 2.0)
    s = -tau * s    # equiv. to s / torch.pow(x,2.0)
    if log: return s
    s = torch.exp(s)
    return s


def posn2vec(M, posn, tau=None):
    """
    Map discrete position to column of matrix by look-up
    or soft position to column of matrix via attention.
    """
    if tau is None:
        return M[:,posn]
    attn = posn2attn(posn, tau, M.shape[-1])
    return attn2vec(attn)


def posn2vec_batch(M, posn, tau=None):
    """
    Map discrete position to column of matrix by look-up
    or soft position to column of matrix via attention
    note: discrete position must be torch.LongTensor,
    soft position must be torch.FloatTensor.
    => output is nbatch x nrow(M)
    """
    if tau is None:
        return M[:,posn].t()
    attn = posn2attn_batch(posn, tau, M.shape[-1])
    val = attn2vec_batch(M, attn)
    return val
# example with discrete positions:
# posn = torch.LongTensor(np.array([0,3,5]))
# u = posn2vec_batch(U, posn)
# print(u.data)


def attn2vec(M, attn):
    """
    Map attention distribution to column of matrix.
    """
    val = torch.mm(M, attn)
    return val


def attn2vec_batch(M, attn):
    """
    Map attention distribution to column of matrix.
    """
    val = torch.mm(M, attn.t()).t()
    #val = torch.einsum('bj,ij->bi', [M, attn])
    return val


def posn2role(posn, tau=None):
    """
    Map soft position to soft role vector.
    """
    R = config.R
    return posn2vec(R, posn, tau)


def posn2role_batch(posn, tau=None):
    """
    Map soft position to soft role vector,
    output is nbatch x drole.
    """
    R = config.R
    return posn2vec_batch(R, posn, tau)


def attn2role_batch(attn):
    R = config.R
    return attn2vec_batch(R, attn)


def attn2succ_batch(attn):
    return attn2vec_batch(S, attn)


def posn2unbind(posn, tau=None):
    """
    Map soft position to soft unbinding vector.
    """
    U = config.U
    return posn2vec(U, posn, tau)


def posn2unbind_batch(posn, tau=None):
    """
    Map soft position to soft unbinding vector,
    output is nbatch x drole.
    """
    U = config.U
    return posn2vec_batch(U, posn, tau)


def attn2unbind_batch(attn):
    """
    Map attention distribution to soft unbinding vector.
    """
    U = config.U
    return attn2vec_batch(U, attn)


def posn2filler(T, posn, tau=None):
    """
    Unbind filler at hard or soft string position.
    """
    u = posn2unbind(posn, tau)
    f = T.mm(u)
    return f


def posn2filler_batch(T, posn, tau=None):
    """
    Unbind filler at hard or soft string position
    for each position i in batch posn
    - get unbinding vector ui by look-up or attention
    - unbind filler fi from tpr Ti with ui
    output is nbatch x nfill.
    """
    u = posn2unbind_batch(posn, tau)
    #print(T.shape, u.shape)
    f = T.bmm(u.unsqueeze(2))
    #print(f.shape)
    f = f.squeeze(-1)
    #print(f.shape)
    return f


def attn2filler_batch(T, attn):
    u = attn2vec_batch(config.U, attn)
    f = T.bmm(u.unsqueeze(2))
    f = f.squeeze(-1)
    return f


def bind_batch(f, r):
    """
    Contruct filler-role bindings
    [https://discuss.pytorch.org/t/batch-outer-product/4025/2]
    output is nbatch x nfill x nrole.
    """
    fr = torch.bmm(f.unsqueeze(2), r.unsqueeze(1))
    return fr


def normalize(X, dim=1):
    """
    Normalize by summing over second dimension
    (NB. does not take absolute value of elements,
    therefore not equivalent to torch.normalize(1,1)).
    todo: relocate
    """
    Y = X / torch.sum(X, dim, keepdim=True)
    return Y


def normalize_length(X, dim=1):
    """
    Normalize to length one.
    todo: relocate
    """
    Y = X / torch.mm(X.t(), X)
    return Y


def bound_batch_old(X):
    """
    Bound columns of each batch within [-1,1].
    todo: relocate
    """
    batch_size, m, n = X.shape
    Y = torch.zeros_like(X)
    ones = torch.ones((1, n))
    for i in range(batch_size):
        Xi = X[i,:,:]
        maxi = torch.max(Xi, 0)[0].view(1, n)
        mini = torch.min(Xi, 0)[0].view(1, n)
        maxi = torch.cat((maxi, -mini, ones), 0)
        maxi = torch.max(maxi, 0)[0].view(1, n)
        Y[i,:,:] = Xi / maxi
    return Y


def bound_batch(X):
    """
    Bound columns of each batch within [-1,1].
    todo: relocate
    """
    batch_size, m, n = X.shape
    ones = torch.ones((batch_size,1,n))
    maxi = torch.max(X, 1)[0].view(batch_size, 1, n)
    mini = torch.min(X,1)[0].view(batch_size, 1, n)
    maxi = torch.cat((maxi, -mini, ones), 1)
    maxi = torch.max(maxi, 1)[0].view(batch_size, 1, n)
    Y = X / maxi
    #delta = torch.sum(torch.sum(Y - rescale_batch(X), 0), 0)
    #print(delta.data[0])
    return Y


def check_bounds(x, min=0.0, max=1.0):
    """
    Check that each value of vector is within bounds.
    todo: relocate
    """
    if np.any(x<min) or np.any(x>max):
        return 0
    return 1
