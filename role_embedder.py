#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
import torch
import randVecs
import numpy as np
import re, sys

class RoleEmbedder():
    def __init__(self, nrole, random_roles=False):
        # role vectors (nrole x drole), and unbinding vectors (nrole x drole),
        # where in the case of tprs U = (R^{-1})
        # note: use transpose of role matrix to facilitate (soft) indexation of columns
        R = torch.eye(nrole)
        U = torch.eye(nrole)
        if random_roles:
            R.data = torch.FloatTensor( randVecs.randVecs(nrole, nrole, np.eye(nrole)) )
            #R.data.normal_() #R.data.uniform_(-1.0, 1.0)
            U.data = torch.FloatTensor(np.linalg.inv(R.data))
            R = R.t()   # xxx

        # successor matrix for localist roles (as in Vindiola PhD)
        # note: torodial boundary conditions
        #Rlocal = torch.eye(drole)
        #S = torch.zeros(drole,drole)
        #for i in range(nrole):
        #    j = i+1 if i<(nrole-1) else 0
        #    S = torch.addr(S, Rlocal[:,j], Rlocal[:,i])

        # test successor matrix
        #if 0:
        #    print(S.data.shape)
        #    p0 = torch.zeros(nrole,1); p0.data[0] = 1.0
        #    for i in range(nrole):
        #        p0 = S.mm(p0)
        #        print(i, '->', p0.data.numpy()[:,0])
        #    sys.exit(0)

        config.drole    = nrole
        config.R        = R
        config.U        = U