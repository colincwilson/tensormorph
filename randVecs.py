#!/usr/bin/env python
# -*- coding: utf-8 -*-
# includes port of dotProducts.m code by Don Mathis.
import numpy as np


# todo: reorder N and dim arguments throughout, make N==dim default
def randVecs(N, dim, sim, lb=-1.0, ub=1.0, scale=False):
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
    M = np.matrix(np.random.rand(dim, N)) # xxx respect bounds at initialization
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


# todo: replace with randSphere
def indepVecs(N, dim, lb=0.0, ub=1.0):
    # xxx only correct for lb = 0, ub = 1
    M = np.random.rand(dim, N)
    return M


# Generate N unit length vectors on the sphere S^dim
# see among others https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
def randSphere(N, dim):
    M = np.random.randn(dim, N)
    M /= np.linalg.norm(M, axis=0)
    M_inv = np.linalg.inv(M).T
    #print (M.T @ M)
    #print (np.round(M.T @ M_inv, 5)) 
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

    print ('randSphere test')
    M = randSphere(5, 5)
    print (M)

if __name__ == "__main__":
    test()
