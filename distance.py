#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .tpr import *
import sys

def sqeuclid(x, Y):
    """
    Squared Euclidean distance between vector x and each column of Y
    """
    # ||x||^2
    x_norm = torch.norm(x, p=2, dim=0)**2.0

    # ||y||^2 for each column in Y
    Y_norm = torch.norm(Y, p=2, dim=0)**2.0

    dist = x_norm + Y_norm - 2.0 * Y.t() @ x
    return dist


def sqeuclid_batch(X, Y):
    """
    Squared Euclidean distance between each column of X and all columns of Y
    see: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    X is N x (m x n), where N is batch index, Y is (m x p), result is N x (p x n)
    todo: simplify by moving m to final dimension
    """
    # ||x||^2 for each column of each batch in X
    # reshaped to N x (n x 1)
    X_norm = torch.norm(X, p=2, dim=1)**2.0
    X_norm = X_norm.unsqueeze(2)

    # ||y||^2 for each column in Y 
    # reshaped to 1 x (1 x p)
    Y_norm = torch.norm(Y, p=2, dim=0)**2.0
    Y_norm = Y_norm.unsqueeze(0).unsqueeze(1)

    # Squared Euclidean distance, with resulting shape N x (n x p),
    # reshaped to N x (p x n) to better match shape of X
    dist = X_norm + Y_norm - 2.0 * torch.matmul(X.transpose(1,2), Y)
    dist = dist.transpose(1,2)
    return dist


def euclid_batch(X, Y):
    """
    Euclidean distance between each column of X and all columns of Y
    """
    dist = sqeuclid_batch(X, Y)
    dist = torch.pow(dist, 0.5)
    return dist


def pairwise_distance(X, W, A=None):
    """
    Pairwise Euclidean distance between fillers of X and patterns of W, 
    optionally with dimension weights A.
    Args:
        TPR X (nbatch, dfill, nrole) in [-1,+1]
        feature values W (nbatch, dfill, npattern) in [-1,+1]
        attention weights A (nbatch, dfill, npattern) in [0,1]
    Returns:
        distance D (nbatch, nrole, npattern) 
    """
    #print (X.shape, W.shape, A.shape)
    D = X.unsqueeze(3) - W.unsqueeze(2)
    D = torch.pow(D, 2.0)
    if A is not None:
        D *= A.unsqueeze(2)
    D = torch.sum(D, 1)
    D = torch.pow(D, 0.5)
    #print (X.shape, W.shape, '->', D.shape)
    return (D)


def cosine_batch(X, Y):
    """
    Cosine similarity between each column of X and all columns of Y.
    X is N x (m x n), where N is batch index, Y is (m x p), result is N x (p x n)
    """
    # ||x|| for each column of each batch in X
    # reshaped to N x (n x 1)
    #X_norm = (X**2.0).sum(1)
    X_norm = torch.norm(X, p=2, dim=1)
    X_norm = X_norm.unsqueeze(2)

    # ||y|| for each column in Y 
    # reshaped to 1 x (1 x p)
    #Y_norm = (Y**2.0).sum(0)
    Y_norm = torch.norm(Y, p=2, dim=0)
    Y_norm = Y_norm.unsqueeze(0).unsqueeze(1)

    # Cosine similarity, with resulting shape N x (n x p), 
    # reshaped to N x (p x n) to better match shape of X
    sim = torch.matmul(X.transpose(1,2), Y)
    sim /= (X_norm * Y_norm)
    sim = sim.transpose(1,2)

    return sim


# test
def main():
    # TPR input
    nbatch, dfill, nrole = 10, 15, 20
    X = torch.rand((nbatch, dfill, nrole))
    print ('X:', X.shape)

    # feature coefficients for each pattern
    npattern = 12
    W = torch.rand((dfill, npattern))
    W = torch.tanh(W)
    print ('W:', W.shape)

    # feature attention for each pattern
    A = torch.rand((dfill, npattern))
    A = torch.sigmoid(A)
    print ('A:', A.shape)

    D = pairwise_distance(X, W, A)
    print ('D:', D.shape)

if __name__ == "__main__":
    main()