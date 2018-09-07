#!/usr/bin/env python
# -*- coding: utf-8 -*-
# todo
# - auto-detect vowels if vowel set not provided
# - allow segment embeddings to be trained (except for delimiter and segment features)

import tpr
from tpr import *
import re, sys

epsilon, stem_begin, stem_end = u'ε', u'⋊', u'⋉'

class SeqEmbedder():
    def __init__(self, syms, vowels=None, nrole=20, random_fillers=False, random_roles=False):
        syms_reg = [x for x in set(syms)] if vowels is None\
                   else [x for x in (set(syms) | set(vowels))]
        syms = [epsilon,] + [stem_begin,] + syms_reg + [stem_end,]
        print 'symbols:', ' '.join([x for x in syms])
        sym2id = { sym: i for i,sym in enumerate(syms) }
        id2sym = { sym2id[x]:x for x in sym2id }
        nfill, nrole, drole = len(syms), nrole, nrole # number of fillers and roles

        # # # # #
        # filler vector matrix (nfill x dfill), one vector per column
        # vectors for regular symbols
        nfill_reg = len(syms_reg)
        F = torch.eye(nfill_reg)
        if random_fillers:
            F.data = torch.FloatTensor(\
                randVecs.randVecs(nfill_reg, nfill_reg, np.eye(nfill_reg))
            )
        # vectors for epsilon and boundary symbols (initialized)
        zero_vec = torch.zeros((F.shape[0], 1))
        F = torch.cat((zero_vec, zero_vec, F, zero_vec), 1)
        # add privative features indicating symbol and delimiters
        sym_ftr         = torch.ones((1,nfill))
        begin_ftr       = torch.zeros((1,nfill))
        end_ftr         = torch.zeros((1,nfill))
        sym_ftr[0,0]    = 0.0   # epsilon is not a regular symbol
        begin_ftr[0,1]  = 1.0   # privative mark for initial word boundary
        end_ftr[0,-1]   = 1.0   # privative mark for final word boundary
        root_ftrs       = torch.cat((sym_ftr, begin_ftr, end_ftr), 0)
        # add consonant/vowel privative feature (if vowels specified)
        if vowels:
            c_ftr = torch.ones(1,nfill)
            v_ftr = torch.zeros(1,nfill)
            c_ftr[0,0] = c_ftr[0,1] = c_ftr[0,-1] = 0.0
            v_ftr[0,0] = v_ftr[0,1] = v_ftr[0,-1] = 0.0
            for vowel in vowels:
                c_ftr[:,sym2id[vowel]] = 0.0
                v_ftr[:,sym2id[vowel]] = 1.0
            root_ftrs = torch.cat((root_ftrs, c_ftr, v_ftr), 0)
        F = torch.cat((root_ftrs, F), 0)
        dfill, nfill = F.shape

        # role vectors (nrole x drole), and unbinding vectors (nrole x drole),
        # where in the case of tprs U = (R^{-1})
        # note: use transpose of role matrix to facilitate (soft) indexation of columns
        R = torch.eye(drole)
        U = torch.eye(drole)
        if random_roles:
            R.data = torch.FloatTensor( randVecs.randVecs(nrole, nrole, np.eye(nrole)) )
            #R.data.normal_() #R.data.uniform_(-1.0, 1.0)
            U.data = torch.FloatTensor(linalg.inv(R.data))
            R = R.t()

        # successor matrix for localist roles (as in Vindiola PhD)
        # note: torodial boundary conditions
        Rlocal = torch.eye(drole)
        S = torch.zeros(drole,drole)
        for i in xrange(nrole):
            j = i+1 if i<(nrole-1) else 0
            S = torch.addr(S, Rlocal[:,j], Rlocal[:,i])

        # test successor matrix
        if 0:
            print S.data.shape
            p0 = torch.zeros(nrole,1); p0.data[0] = 1.0
            for i in xrange(nrole):
                p0 = S.mm(p0)
                print i, '->', p0.data.numpy()[:,0]
            sys.exit(0)

        if 0:   # inspect F, R, U
            print np.round(F.data.numpy(), 3)
            print np.round(R.data.numpy(), 3)
            print np.round(U.data.numpy(), 3)
            sys.exit(0)

        self.syms, self.vowels, self.sym2id, self.id2sym = syms, vowels, sym2id, id2sym
        self.F, self.R, self.U = F, R, U
        self.random_fillers, self.random_roles = random_fillers, random_roles
        self.nfill, self.nrole, self.dfill, self.drole = nfill, nrole, dfill, drole

    # get filler vector for symbol
    def sym2vec(self, x):
        sym2id, F = self.sym2id, self.F
        return F.data[:,sym2id[x]]

    # map string to vector of indices
    # (input must be space-separated)
    def string2idvec(self, x, delim=True):
        sym2id = self.sym2id
        y = self.string2delim(x) if delim else x
        y = [sym2id[yi] for yi in y.split(u' ')]
        return y

    # map string to tpr
    # input must be space-separated
    def string2tpr(self, x, delim=True):
        sym2id, F, R = self.sym2id, self.F, self.R
        y = string2delim(x) if delim else x
        y = y.split(' ')
        n = len(y)
        if n >= self.nrole:
            print 'string2tpr error: string length longer than nrole for string', x
            return None
        Y = torch.zeros(self.dfill, self.drole)
        for i in xrange(n):
            #print i, y[i], sym2id[y[i]]
            Y += torch.ger(F.data[:,sym2id[y[i]]], R.data[:,i]) # outer product
        return Y

    # mark up output with deletion and (un)pivot indicators
    def idvec2string(self, x, copy=None, pivot=None):
        id2sym = self.id2sym
        segs = [id2sym[id] for id in x]
        y = ' '.join(segs)
        y = re.sub(u'⋉.*', u'⋉', y)
        segs = y.split(' ')
        if copy is not None:
            segs = [u'⟨'+x+u'⟩' if (i<len(copy) and copy[i]<0.5) else x for i,x in enumerate(segs)]
        if pivot is not None:
            segs = [x+u' •' if (i<len(pivot) and pivot[i]>0.5) else x for i,x in enumerate(segs)]
        y = ' '.join(segs)
        return y

# separate elements of string with spaces
def string2sep(x):
    x = ' '.join([xi for xi in x])
    return x

# convert strings to vectors or tensor-product representations
# add word delimiters; input must be space-separated
def string2delim(x):
    val = [stem_begin,] + [xi for xi in x.split(' ')] + [stem_end,]
    val = ' '.join(val)
    return val
