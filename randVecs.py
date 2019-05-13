#!/usr/bin/env python
# -*- coding: utf-8 -*-
# port of dotProducts.m code by Don Mathis.
import numpy as np

def randVecs(N, dim, sim, lb=-1.0, ub=1.0, scale=True):
    """Create random vectors (columns) with specified similarity values.

    Args:
        N (int): Number of random vectors to create.
        dim (int): Dimensionality of each random vector.
        sim (matrix): Specified similarity matrix (N x N).

    Returns:
        dim x N matrix with random vectors in rows.
    """
    dpMatrix = sim * np.ones((N,N)) + (1-sim) * np.identity(N)
    M = randVecs2(N, dim, dpMatrix, lb, ub, scale)
    return M

def randVecs2(N, dim, dpMatrix, lb=-1.0, ub=1.0, scale=True):
    M = np.matrix(np.random.rand(dim,N)) # xxx respect bounds at initialization
    step0 = 0.1
    tol = 1e-6
    maxIts = 50000
    for i in range(maxIts):
        inc = np.dot(M, np.dot(M.T,M) - dpMatrix)
        step1 = 0.01 / np.max(np.abs(inc))
        step = np.min((step0,step1))
        M = M - step * inc
        M[M<lb] = lb; M[M>ub] = ub # xxx bound values to [lb,ub]
        #M[M<-1.0/float(N)] = -1.0/float(N); M[M>1.0/float(N)] = 1.0/float(N)
        maxDiff = np.max(np.abs(np.dot(M.T,M) - dpMatrix))
        if maxDiff <= tol:
            if scale: # scale values by number of rand vecs
                return 1.0/float(N) * M
            return M
    print('randVecs2: Failed to find solution to tolerance in specified iterations')
    return M

def indepVecs(N, dim, lb=0.0, ub=1.0):
    # xxx only correct for lb = 0, ub = 1
    M = np.random.rand(dim, N)
    return M

def test():
    M = randVecs(5, 5, np.identity(5), scale=False)
    print(M)
    sim = np.round(np.dot(M.T, M), 5)
    print(sim)
    Minv = np.linalg.inv(M)
    print (Minv)

    M = indepVecs(5, 5)
    print (M)
    Minv = np.linalg.inv(M)
    print (Minv)
    print (M[:,0] * M[:,1])

if __name__ == "__main__":
    test()
