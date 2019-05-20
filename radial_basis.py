#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
from .tpr import *

class GaussianPool(nn.Module):
    def __init__(self, n, tau=None):
        super(GaussianPool, self).__init__()
        self.n = n # number of units in pool
        self.mu = torch.arange(n).type(torch.FloatTensor)
        if tau: # attentional precision
            self.tau = Parameter(torch.ones(1)*tau)
        else:
            self.tau = Parameter(torch.zeros(1)) # .normal_(1.0, 0.1))

    # map batch of soft positions to attention distributons over discrete positions
    # => output is nbatch x drole
    def forward(self, posn):
        mu, tau = self.mu, torch.exp(self.tau) + config.tau_min # apply relu6 to tau? #hardtanh(self.tau, 0.0, 10.0)
        attn = -tau*torch.pow(posn - mu, 2.0) # distance between posn and each rbf center
        attn = log_softmax(attn, 1) # normalize in log domain
        attn = torch.exp(attn) # convert to prob distribution
        assert(not np.any(np.isnan(attn.data.numpy()))), 'attention value is nan'\
            + str(attn.data.numpy())\
            +' from '+ str(posn.data.numpy())
        assert(np.all(attn.data.numpy()<=1.0)),\
           'attention value greater than 1'
        return attn


# smooth probability distribution over ordinal positions with
# ~ Gaussian convolution [xxx can this be implemented directly?]
class GaussianPool2D(nn.Module):
    def __init__(self, n, tau=None):
        super(GaussianPool2D, self).__init__()
        self.n   = n # number of ordinal positions
        # neg squared distance matrix on ordinal positions
        mu = torch.arange(n).type(torch.FloatTensor)
        self.M = mu.view(n,1) - mu.view(1,n)
        self.M = -torch.pow(self.M, 2.0)
        if tau: # attentional precision
            self.tau = Parameter(torch.ones(1) * tau)
        else:
            self.tau = Parameter(torch.FloatTensor(1).uniform_(1.0e-1, 1.0))

    def forward(self, b):
        n, M, tau = self.n, self.M, relu6(self.tau)
        dist = torch.exp(log_softmax(tau * M, 0))
        beta = b.mm(dist)
        beta = normalize(beta, 1, 1)
        return beta


# map soft position to attention distribution over discrete positions
def posn2attn(posn, tau, n=None):
    mu   = torch.arange(0,n).type(torch.FloatTensor)
    attn = posn - mu
    attn = -tau*torch.pow(attn, 2.0)   # rbf scores
    attn = log_softmax(attn, 0)    # log_softmax
    attn = torch.exp(attn)             # convert to prob distrib
    return attn


def test():
    n = 10
    tau = 1.0
    nbatch = 2
    gp = GaussianPool(n, tau)
    b = torch.zeros(nbatch,1).type(torch.FloatTensor)
    print(gp(b))

    gp2 = GaussianPool2D(n, tau)
    print(gp2.M)
    b = torch.FloatTensor(nbatch,n)
    b.data[:] = 0.0; b.data[0,0] = 1.0; b.data[0,4] = 0.1; b.data[1,9] = 1.0
    print(gp2(b))

if __name__ == "__main__":
    test()
