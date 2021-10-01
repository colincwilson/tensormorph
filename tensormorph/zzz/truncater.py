# -*- coding: utf-8 -*-

import config
from tpr import *
#from radial_basis import GaussianPool
from pivoter2 import BiPivoter
import distributions as distrib
import mcmc


class BiTruncater(nn.Module):
    """
    Combine results of truncating LR-> and <-RL,
    output is a copy vector (nbatch x n)
    maybe: prevent deletion of end delimiter (for use as pivot)
    """
    def __init__(self, dcontext=1, nfeature=5):
        super(BiTruncater, self).__init__()
        # Select scan direction
        self.pivoter = BiPivoter(dcontext, nfeature)
        self.context2delete_prefix = Linear(dcontext, 2)
        self.context2delete_begin = Linear(dcontext, 2)
        self.context2delete_end = Linear(dcontext, 2)
        if dcontext == 1: # remove redundant weight parameters
            self.context2delete_prefix.weight.detach_()
            self.context2delete_begin.weight.detach_()
            self.context2delete_end.weight.detach_()
        self.stochastic = []

    def forward(self, form, context):
        # Identify point at which truncation begins / ends
        pivot, _ = self.pivoter(form, context)

        # Mark positions up to pivot (inclusive)
        # xxx make deterministic as in pivoter
        nbatch, nrole = form.shape[0], config.nrole
        mask = hardtanh(form[:,0,:], 0.0, 1.0)
        mark = torch.zeros(nbatch, nrole)
        flag = torch.ones(nbatch)
        for i in range(nrole):
            mark[:,i] = flag
            flag = pivot[:,i] * 0.0 + (1 - pivot[:,i]) * flag
        #mark = mark.detach() # xxx block backprop
        #print(mark.shape, mask.shape); sys.exit(0)
        mark_prefix = mark * mask
        mark_suffix = (1.0 - mark) * mask

        # Truncate marked -or- unmarked positions -or- nothing
        delete_prefix = distrib.rsample(
            self.context2delete_prefix(context))
        delete = bscalmat(delete_prefix[:,0], mark_prefix) \
               + bscalmat(delete_prefix[:,1], mark_suffix)
        copy = 1.0 - delete

        # Special handling of begin/end delimiters 
        # (see phon_features.py for conventions on one-hot
        # coding of begin and end boundary symbols)
        delete_begin = distrib.rsample(
            self.context2delete_begin(context))
        delete_end = distrib.rsample(
            self.context2delete_end(context))
        copy_begin = 1.0 - bscalmat(delete_begin[:,1], form[:,1,:])
        copy_end = 1.0 - bscalmat(delete_end[:,1], form[:,2,:])
        copy = copy * copy_begin # xxx move to log space
        copy = copy * copy_end

        # Mask out epsilon fillers (i.e., always delete them)
        # (see phon_features.py for convention on one-hot
        # coding of epsilon vs. non-epsilon distinction)
        mask = hardtanh(form[:,0,:], 0.0, 1.0) # xxx see tpr.epsilon_mask
        copy = copy * mask
        #print(copy_begin.shape, copy_end.shape)
        #print(copy.shape); sys.exit(0)
        return copy


    def init(self):
        for layer in [self.context2delete_prefix,
                      self.context2delete_begin,
                      self.context2delete_end]:
            layer.weight.data.fill_(0.0)
            layer.bias.data.fill_(0.0)
            layer.bias.data[0] = 1.0
        # Register stochastic params
        self.stochastic = [
            mcmc.StochasticParameter(
                self.context2delete_prefix.bias,
                distrib.SphericalNormal(n=2), #distrib.Discrete(n=2),
                distrib.Discrete(n=2)),
            mcmc.StochasticParameter(
                self.context2delete_begin.bias,
                distrib.SphericalNormal(n=2), #distrib.Discrete(torch.tensor([0.75, 0.25])),
                distrib.Discrete(n=2)),
            mcmc.StochasticParameter(
                self.context2delete_end.bias,
                distrib.SphericalNormal(n=2),
                None)
        ]


    def init_(self, delete_before=False, delete_prefix=False, delete_end=False, bias=10.0, clamp=True):
        for m in [self.delete_prefix,
                  self.delete_begin, self.delete_end]:
            for p in m.parameters():
                p.data.fill_(0.0)
                p.requires_grad = (not clamp)
        self.delete_prefix.bias.data.fill_(
            (1.0 if delete_prefix else -1.0) * bias )
        self.delete_begin.bias.data.fill_(
            (1.0 if delete_begin else -1.0) * bias )
        self.delete_end.bias.data.fill_(
            (1.0 if delete_end else -1.0) * bias )
        self.pivoter.init()