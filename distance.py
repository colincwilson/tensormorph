#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr
from tpr import *

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