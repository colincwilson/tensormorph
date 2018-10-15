#!/usr/bin/env python
# -*- coding: utf-8 -*-

# port of dotProducts.m code by Don Mathis.
import numpy as np

def randVecs(N, dim, sim):
    """Create random vectors (columns) with specified similarity values.

    Args:
        N (int): Number of random vectors to create.
        dim (int): Dimensionality of each random vector.
        sim (matrix): Specified similarity matrix (N x N).

    Returns:
        dim x N matrix with random vectors in rows.
    """
    dpMatrix = sim * np.ones((N,N)) + (1-sim) * np.identity(N)
    M = randVecs2(N, dim, dpMatrix)
    return M

def randVecs2(N, dim, dpMatrix):
    M = np.matrix(np.random.rand(dim,N)) # xxx respect bounds at initialization
    step0 = 0.1
    tol = 1e-6
    maxIts = 50000
    for i in range(maxIts):
        inc = np.dot(M, np.dot(M.T,M) - dpMatrix)
        step1 = 0.01 / np.max(np.abs(inc))
        step = np.min((step0,step1))
        M = M - step * inc
        M[M<-1.0] = -1.0; M[M>1.0] = 1.0 # xxx restrict values to [-1,1]
        maxDiff = np.max(np.abs(np.dot(M.T,M) - dpMatrix))
        if maxDiff <= tol:
            return M
    print('randVecs2: Failed to find solution to tolerance in specified iterations')
    return M

def test():
    M = randVecs(5, 6, np.identity(5))
    print(M)
    sim = np.round(np.dot(M.T, M), 5)
    print(sim)

if __name__ == "__main__":
    test()
