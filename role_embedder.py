#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
from .randVecs import randVecs
import torch
import numpy as np
import re, sys

class RoleEmbedder():
    def __init__(self, nrole, randVecs_kwargs=None):
        """
        Create (currently only LR->) role vectors (drole x nrole) and 
        corresponding unbinding vectors (also drole x nrole), where for 
        exact TPR unbinding U = inv(R).
        ??? note: use transpose of role matrix to facilitate (soft) indexation of columns
        """
        if randVecs_kwargs is None:
            R = np.eye(nrole)
        else:
            randVecs_kwargs['n'] = nrole
            randVecs_kwargs['dim'] = nrole
            R = randVecs(**randVecs_kwargs)
        R = torch.FloatTensor(R)
        R.requires_grad = False

        U = torch.FloatTensor(np.linalg.inv(R)).t()
        U.requires_grad = False

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