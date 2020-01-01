#!/usr/bin/env python
# -*- coding: utf-8 -*-
# includes port of dotProducts.m code by Don Mathis.
import numpy as np


def randvecs(n=None, dim=None, sim=None, sphere=False, nonnegative=False, scale=1.0):
    """Creates random vectors using one of the methods below.
    Args:
        n (int): Number of random vectors to create.
        dim (int, optional): Dimensionality of reach random vector (== n if omitted).
        sim (matrix, optional): Specified similarity matrix (n x n).
        sphere (boolean, optional): Vectors randomly generated on the unit sphere.
        nonnegative (boolean, optional): All vector elements constrained to be nonnegative.
        scale (float, optional): Multiply all vector elements by 1/scale.
    """
    if dim is None:
        dim = n
    if sim is not None:
        M = randvecs_sim(n, dim, sim)
    elif sphere is not None:
        M = randvecs_sphere(n, dim, nonnegative)
    M /= float(scale)
    return M


def randvecs_sim(n, dim, sim, lb=-1.0, ub=1.0, scale=False):
    """Create random vectors (columns) with specified similarity values.
    Args:
        n (int): Number of random vectors to create.
        dim (int): Dimensionality of each random vector.
        sim (matrix): Specified similarity matrix (n x n).

    Returns:
        dim x n matrix with random vectors in rows.
    """
    dpMatrix = sim * np.ones((n,n)) + (1.0-sim) * np.identity(n)
    M = randvecs_sim2(n, dim, dpMatrix, lb, ub, scale)
    return M


def randvecs_sim2(n, dim, dpMatrix, lb=-1.0, ub=1.0, scale=True):
    M = np.matrix(np.random.rand(dim, n)) # xxx respect bounds at initialization
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
    print('randvecs_sim2: Failed to find solution to tolerance in specified iterations')
    return M


def randvecs_sphere(N, dim, nonnegative=False):
    """
    Generate n unit-length vectors on surface of sphere S^dim, 
    optionally requiring all elements to be non-negative, 
    and verify linear independence by inversion.
    # see among others https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
    """
    M = np.random.randn(dim, N)
    if nonnegative:
        M = np.abs(M)
    M /= np.linalg.norm(M, axis=0)
    M_inv = np.linalg.inv(M)
    #print (M.T @ M)
    #print (np.round(M.T @ M_inv.T, 5)) 
    return M


def test():
    print ('randvecs_sim')
    n = 5
    M = randvecs(**{'n':n, 'dim':n, 'sim':np.identity(n)})
    print(M)
    sim = np.round(np.dot(M.T, M), 5)
    print(sim)
    Minv = np.linalg.inv(M)
    print (Minv)

    print ('randvecs_sphere -- general')
    M = randvecs(**{'n':n, 'dim':n, 'sphere':True})
    #M = rand_vecs(n, n, sphere=True)
    print (M)

    print ('randvecs_sphere -- nonnegative')
    #M = rand_vecs(n, n, sphere=True, nonnegative=True)
    M = randvecs(**{'n':n, 'dim':n, 'sphere':True})
    print (M)


if __name__ == "__main__":
    test()