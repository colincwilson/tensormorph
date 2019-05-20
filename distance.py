#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .tpr import *

# calculate squared Euclidean distance between fillers of X and columns of F
# see: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
# X is N x (m x n), where N is batch index, Y is (m x p)
def euclid_squared_batch(X, Y):
    # get ||x||^2 for each column of each batch in X
    # and reshape to N x (n x 1)
    X_norm = (X**2.0).sum(1)
    X_norm = X_norm.unsqueeze(2)

    # get ||y||^2 for each column in Y 
    # and reshape to 1 x (1 x p)
    Y_norm = (Y**2.0).sum(0)
    Y_norm = Y_norm.unsqueeze(0).unsqueeze(1)

    # compute distance, result is N x (n x p),
    # and reshape to N x (p x n) to better match shape of X
    dist = X_norm + Y_norm - 2.0 * torch.matmul(X.transpose(1,2), Y)
    dist = dist.transpose(1,2)
    return dist

def euclid_batch(X, Y):
    dist = euclid_squared_batch(X, Y)
    dist = torch.pow(dist, 0.5)
    return dist

# given: TPR X (nbatch, dfill, nrole) in [-1,+1],
# feature values W (nbatch, dfill, npattern) in [-1,+1],
# attention weights A (nbatch, dfill, npattern) in [0,1]
# return: pairwise Euclidean distance D (nbatch, nrole, npattern) 
# between fillers of X and patterns of W, optionally weighted by A
def pairwise_distance(X, W, A=None):
    #print (X.shape, W.shape, A.shape)
    D = X.unsqueeze(3) - W.unsqueeze(2)
    D = torch.pow(D, 2.0)
    if A is not None:
        D *= A.unsqueeze(2)
    D = torch.sum(D, 1)
    D = torch.pow(D, 0.5)
    #print (X.shape, W.shape, '->', D.shape)
    return (D)


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