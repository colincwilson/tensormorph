#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
from .randvecs import randvecs
import torch
import numpy as np
import re, sys


class RoleEmbedder():
    def __init__(self, nrole, randvec_params=None):
        """
        Create (currently only LR->) role vector matrix (drole x nrole) and 
        corresponding unbinding vector matrix (also drole x nrole), where for 
        exact TPR unbinding U = inv(R).
        ??? note: use transpose of role matrix to facilitate (soft) indexation of columns
        """
        if randvec_params is None:
            R = np.eye(nrole)
        else:
            randvec_params['n'] = nrole
            randvec_params['dim'] = nrole
            R = randvecs(**randvec_params)
        R = torch.tensor(R, requires_grad=False, dtype=torch.float)
        U = torch.tensor(np.linalg.inv(R), requires_grad=False).t()

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

        self.nrole = R.shape[1]
        self.drole = R.shape[0]
        self.R = R
        self.U = U