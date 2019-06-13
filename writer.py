#!/usr/bin/env python
# -*- coding: utf-8 -*-

# xxx todo: track cumulative attention to each role (position) within this class, 
# taking into account deletion (~ zero-write) steps:
# ex. for role i, a_i(t)*1 + a_i(t+1)*1 + a_i(t+2)*0 should have total a_i(t) + a_i(t+1)
# - An alternative method of normalization would be to divide each elt 
# by its absolute value, i.e., apply sign (signum) function: 
# sign(x) = x / |x| = |x| / x -- but without thresholding this would explode 
# values close to 0 onto {-1,+1}
# - Another alternative applies hardtanh with boundaries [-1, +1]. This differs 
# from cumulative-attention normalization, ex. (.95*1)/.95 = 1 vs. hardtanh(.95*1) = .95

from .environ import config
from .tpr import *

# todo: make version for localist roles
class Writer(nn.Module):
    def __init__(self, node='writer'):
        super(Writer, self).__init__()
        self.Y, self.attn_total = None, None
        self.node = node


    # Add a soft symbol/role binding to tpr Y
    #   a is soft index into morphs (M0, M1)
    #   b0, b1 are soft positions within M0, M1
    #   c is soft position within output Y
    def forward(self, M0, M1, alpha, beta0, beta1, omega, delta0, delta1):
        Y, attn_total = self.Y, self.attn_total
        theta = alpha[:,0].unsqueeze(1)       # attention to morph M0 or M1
        x0    = attn2filler_batch(M0, beta0)  # soft filler in M0 at soft index beta0
        x1    = attn2filler_batch(M1, beta1)  # soft filler in M1 at soft index beta1
        x     = theta * delta0 * x0 + (1.0-theta) * delta1 * x1 # convex combo of soft fillers
        r     = attn2role_batch(omega)        # soft role corresponding to soft index omega
        Y     = torch.baddbmm(Y, x.unsqueeze(2), r.unsqueeze(1))   # accumulate binding into output
        attn_total = attn_total +\
            (theta * delta0 + (1.0-theta) * delta1) * omega # accumluate attention to output roles
        self.Y, self.attn_total = Y, attn_total

        if config.recorder is not None:
            config.recorder.update_values(self.node, {
                'alpha':alpha,
                'beta0':beta0,
                'beta1':beta1,
                'omega':omega,
                'delta0':delta0,
                'delta1':delta1,
                'output':Y
            })

        return Y


    def normalize(self):
        """
        Normalize (divide) the gradient filler in each role of the output Y 
        by the total attention to that role accumulated over writing steps.
        xxx assumes localist roles
        """
        Y, attn_total = self.Y, self.attn_total
        #print Y.shape, attn_total.shape
        eps = 1.0e-10
        Z = Y / (attn_total.unsqueeze(1) + eps)
        self.Y = Z
        return Z


    def output(self):
        """
        Final result of writing
        """
        return self.Y


    def init(self, nbatch):
        """
        Initialize output and cumulative role attention.
        """
        self.Y = torch.zeros(nbatch, config.dfill, config.drole, requires_grad=True)
        self.attn_total = torch.zeros(nbatch, config.drole)
