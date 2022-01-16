# -*- coding: utf-8 -*-

# Provides batch binding and unbinding (i.e., query) operations on TPRs
# and various convenience methods (e.g., normalization, sequence masking).
# Conventions:
# * Operations apply to batches unless otherwise indicated
# * Batch corresponds to *first* index in all inputs and outputs
#   (see https://discuss.pytorch.org/t/how-to-repeat-a-vector-batch-wise)
# todo: convert ops to einsum

import os, pickle, re, sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch import \
    atanh, cumsum, einsum, exp, log, logsumexp, optim, sigmoid, tanh
from torch.nn import \
    Sequential, LSTM, GRU, Linear, Tanh, Parameter
from torch.nn.functional import \
    hardtanh, linear, log_softmax, relu, relu6, logsigmoid, softmax, \
    softplus, gumbel_softmax, threshold, threshold_, cosine_similarity
from einops import rearrange
#from matcher import match_pos
#from torch.distributions.uniform import Uniform
#from torch.distributions.normal import Normal
#from torch.distributions.bernoulli import Bernoulli
#from torch.distributions.categorical import Categorical
#from torch.distributions.geometric import Geometric
#from torch.distributions.multivariate_normal import MultivariateNormal
#from torch.distributions.one_hot_categorical import OneHotCategorical
#from torch.distributions.relaxed_categorical \
#    import RelaxedOneHotCategorical
import config


def bdot(X, Y):
    """
    Batch dot (inner) product
    [https://discuss.pytorch.org/t/dot-product-batch-wise/9746/11]
    """
    #Z = (X * Y).sum(-1, keepdim=True)
    Z = einsum('b i, b i -> b', X, Y).unsqueeze(-1)
    #print(X.shape, Y.shape, Z.shape); sys.exit(0)
    return Z


def bmatvec(X, v):
    """
    Batch matrix-vector multiplication 
    """
    val = torch.matmul(X, v.unsqueeze(-1))
    val = val.squeeze(-1)
    #print(X.shape, v.shape, val.shape); sys.exit(0)
    # val = einsum('b...j,bj->b...', X, v) # xxx fix me
    #val = X * v.unsqueeze(1)
    #val = torch.sum(val,-1)
    return val


def bscalmat(a, X):
    """
    Multiply one- or two-dim X[i] by scalar a[i] 
    for each batch member i
    """
    if len(X.shape) == 2:
        val = a.view(-1, 1) * X
    elif len(X.shape) == 3:
        val = a.view(-1, 1, 1) * X
    return val


def bind(X, f, r):
    """
    Add filler/role binding (outer product) to X
    """
    X = X + torch.bmm(f.unsqueeze(2), r.unsqueeze(1))
    return X


def unbind(X, r):
    """
    Unbind/query filler from X using discrete or gradient role r
    """
    f = bmatvec(X, r)
    return f


def attn_bind(X, f, attn):
    """
    Bind filler f into X using attention distribution attn over roles
    """
    # Map attention distribution to role
    r = bmatvec(config.R, attn)
    # Bind filler to role
    X = bind(X, f, r)
    return X, r


def attn_unbind(X, attn):
    """
    Unbind/query filler from X using attention distribution attn over roles
    """
    if len(X.shape) > 2:
        # Map attention distribution to role dual
        u = bmatvec(config.U, attn)
        # Unbind filler
        return unbind(X, u)
    if len(X.shape) > 1:
        # Unbind entry from vector
        # (assumes localist repn)
        return bdot(X, attn)
    return None


def distrib2local(X):
    """
    Convert format of X from distributed roles to localist roles
    """
    if not config.random_roles:
        return X
    Xloc = X @ config.U
    return Xloc


def local2distrib(Xloc):
    """
    Convert format of X from localist roles to distributed roles
    """
    if not config.random_roles:
        return Xloc
    X = Xloc @ config.R.t()
    return X


def unbind_ftr(X, start=0, end=None):
    """
    Unbind features with indices [start ... end) from X [nbatch x nftr x nrole]
    """
    Xloc = distrib2local(X)
    if end is None:
        Xftr = Xloc[:, start]
    else:
        Xftr = Xloc[:, start:end]
    return Xftr


def shift1(X, k=0):
    """
    Shift along role dimension of X [nbatch x nftr x nrole] 
    forward (lag/delay) or backward (lead/advance) by 
    one position, padding with zeros
    (assumes localist roles)
    """
    if k > 0:
        # Shift forward (lag/delay) by one position
        Xlag = X @ config.Mlag
        return Xlag
    elif k < 0:
        # Shift backward (lead/advance) by one position
        Xlead = X @ config.Mlead
        return Xlead
    return X  # Identity


def shift(X, k=0):
    """
    Shift along role dimension of X [nbatch x nftr x nrole] 
    forward (lag/delay) or backward (lead/advance) by 
    k positions, padding with zeros
    (assumes localist roles)
    cf. torch.roll, which is toroidal
    """
    if k > 0:
        # Shift forward (lag/delay) by k positions
        shape = X.shape[:-1] + (k,)
        pad = torch.zeros(shape, device=config.device)
        Xlag = torch.cat([pad, X[..., :-k]], -1)
        return Xlag
    elif k < 0:
        # Shift backward (lead/advance) by k positions
        k = -k
        shape = X.shape[:-1] + (k,)
        pad = torch.zeros(shape, device=config.device)
        Xlead = torch.cat([X[..., k:], pad], -1)
        return Xlead
    return X  # Identity


def hardtanh0(X, eps=1.0e-5):
    """
    Apply hardtanh with bounds [0.0+eps, 1.0-eps] 
    where epsilon is close to zero; approximation of 
    relu(hardtanh(.)) that keeps outputs within (0,1)
    """
    return hardtanh(X, min_val=eps, max_val=1.0 - eps)


def hardtanh1(X, eps=1.0e-5):
    """
    Apply hardtanh with bounds [-1.0+eps, 1.0-eps]
    where epsilon is close to zero; approximation of 
    hardtanh(.) that keeps outputs within (-1.0, +1.0)
    """
    return hardtanh(X, min_val=-1.0 + eps, max_val=1.0 - eps)


def threshold0(X, theta=0.5, eps=1.0e-5):
    """
    Heaviside/step function with threshold
    xxx add negative part
    """
    Y = X.clone().detach()
    Y = hardtanh0(sigmoid(5.0 * (Y - theta)), eps)
    return Y


def epsilon_mask(X):
    """
    Gradient mask of fillers that are epsilon in tanh(X).
    Returns log mask; apply exp() to output for values in (0,1).
    ref: on masking in general see 
    torch.nn.modules.transformer.generate_square_subsequent_mask
    """
    sym_ftr = unbind_ftr(X, 0)
    mask = logsigmoid(5.0 * (sym_ftr - 1.0))
    return mask


def apply_mask(X, mask=None):
    """
    Apply mask along role dimension of X [nbatch x nftr x nrole]. 
    All mask elements are assumed to be in (0,1).
    (assumes localist roles)
    """
    if mask is None:
        return X
    if len(mask.shape) < len(X.shape):
        mask = mask.unsqueeze(1)  # Broadcast over features
    return X * mask


def np_round(X, decimals=2):
    """
    Return X as rounded numpy matrix
    """
    return np.round(X.detach().numpy(), decimals)


# xxx move to role_embedder
def rbf(x, mu, tau, log=False):
    """
    Gaussian radial basis function
    (note: does not include normalization constant)
    """
    s = -tau * torch.pow(x - mu, 2.0)
    if log:
        return s
    s = exp(s)
    return s


# xxx move to role_embedder
def scaledlognorm(x, mu, tau=torch.FloatTensor([
    1.0,
]), log=False):
    """
    Botvinick-Watanabe (2007) scaled log normal function
    note: positions begin at 0 (cf. 1 in the original paper),
    therefore add 1 before taking logs ... wlog ;)
    """
    s = torch.pow(log(x + 1.0) - log(mu + 1.0), 2.0)
    s = -tau * s  # equiv. to s / torch.pow(x,2.0)
    if log:
        return s
    s = exp(s)
    return s