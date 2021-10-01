# -*- coding: utf-8 -*-

import config
from randVecs import randvecs
import torch
import torch.nn.functional as F
import numpy as np
import re, sys


class RoleEmbedder():

    def __init__(self, nrole, randvec_params=None):
        """
        Create (currently only LR->) role vector matrix (drole x nrole) 
        and corresponding unbinding vector matrix (also drole x nrole), 
        where for exact TPR unbinding U = inv(R).
        ??? note: use transpose of role matrix to 
        facilitate (soft) indexation of columns
        """
        verbosity = 0

        if randvec_params is not None:
            # Random roles
            randvec_params['n'] = nrole
            randvec_params['dim'] = nrole
            R_ = randvecs(**randvec_params)
        elif config.distrib_roles:
            # Distributed Gaussian position roles
            mu, tau = torch.arange(nrole), 1.0
            R_ = [-tau * torch.pow(mu - float(i), 2.0) for i in range(nrole)
                 ]  # Squared dists from centers
            R_ = [torch.exp(r) for r in R_]  # exp(dists)
            R_ = [F.normalize(r, p=2, dim=0) for r in R_]  # Unit lengths
            R_ = torch.stack(R_, 1)  # Stack role vecs in columns
        else:
            # Orthogonal, one-hot roles
            R_ = np.eye(nrole)

        # Role vectors in columns
        R = torch.tensor(R_,
                         requires_grad=False,
                         dtype=torch.float,
                         device=config.device)

        # Unbinding (role dual) vectors in columns
        U = torch.tensor(np.linalg.inv(R_).T,
                         requires_grad=False,
                         dtype=torch.float,
                         device=config.device)

        if verbosity > 0:
            print(f'role vectors:\n{np.round(R.data.numpy(), 2)}')
            print(f'unbinding vectors:\n{np.round(U.data.numpy(), 2)}')
            I = R.t() @ U
            print(np.round(I.data.numpy(), 2))
            S = R.t() @ R
            print(np.round(S.data.numpy(), 2))
            sys.exit(0)

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
