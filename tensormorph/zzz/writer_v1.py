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

from environ import config
from tpr import *

class Writer(nn.Module):
    """
    Accumulate soft filler/role bindings into output tpr
    todo: make version specialized for localist roles
    xxx make attn_total a persistent buffer or detach?
    """
    def __init__(self):
        super(Writer, self).__init__()
        self.Y, self.attn_total = None, None
        self.attn_total2 = None

    def forward(self, M0, M1, alpha, beta0, beta1, omega, delta0, delta1):
        """
        Add soft filler/role binding to tpr Y
        - M0, M1 are tprs for stem and affix morphs
        - alpha is distribution over morphs [stem, affix]
        - beta0, beta1, omega are distibutions over 
          ordinal positions [0...] in stem, affix, output
        - delta0, delta1 are copy vectors for stem, affix
        """
        if self.Y is None:
            self.Y = torch.zeros_like(M0, requires_grad=True)
            self.attn_total = torch.zeros(M0.shape[0], M0.shape[2])
            self.attn_total2 = torch.zeros(M0.shape[0], M0.shape[2])
        Y, attn_total = self.Y, self.attn_total
        attn_total2 = self.attn_total2

        # Attention to morph M0
        theta = alpha[:,0].unsqueeze(1)
        # Soft filler in M0 at soft index beta0
        x0 = attn2filler_batch(M0, beta0)
        # Soft filler in M1 at soft index beta1
        x1 = attn2filler_batch(M1, beta1)
        # Attention-weighted combo of soft fillers in M0 and M1
        x = theta * delta0 * x0 + (1.0 - theta) * delta1 * x1
        # Soft role corresponding to soft index omega
        r = attn2role_batch(omega)
        # Update tpr with filler/role binding
        Y = torch.baddbmm(Y, x.unsqueeze(2), r.unsqueeze(1))
        # Accumulate attention to output roles
        # xxx distinguish between epsilon and non-epsilon writes?
        attn_total = attn_total + \
            (theta * delta0 + (1.0 - theta) * delta1) * omega
        attn_total2 = attn_total2 + omega
        self.Y, self.attn_total = Y, attn_total
        self.attn_total2 = attn_total2

        if config.recorder is not None:
            config.recorder.update_values(self.node, {
                'alpha': alpha,
                'beta0': beta0,
                'beta1': beta1,
                'omega': omega,
                'delta0': delta0,
                'delta1': delta1,
                'output': Y
            })

        return Y

    def normalize(self):
        """
        Normalize (divide) the gradient filler in each role of the output Y 
        by the total attention to that role accumulated over writing steps.
        xxx assumes localist roles
        xxx how to better keep output elements within [-1,+1]? hardtanh?
        """
        if 0:
            Y, attn_total = self.Y, self.attn_total
            #print Y.shape, attn_total.shape
            eps = 1.0e-10
            Z = Y / (attn_total.unsqueeze(1) + eps)
            self.Y = Z
        elif 0:
            Z = hardtanh(self.Y, -1.0, 1.0)
            Z = torch.tanh(3.0 * self.Y)
            self.Y = Z
        else: # no normalization
            Z = self.Y
        return Z

    def output(self):
        """
        Final result of writing
        xxx use normalize() instead
        """
        return self.Y

    def init(self, nbatch):
        """
        Initialize output and cumulative role attention
        """
        self.Y, self.attn_total = None, None
        self.attn_total2 = None
        #self.Y = torch.zeros(nbatch, config.dfill, config.drole, requires_grad=True)
        #self.attn_total = torch.zeros(nbatch, config.drole)