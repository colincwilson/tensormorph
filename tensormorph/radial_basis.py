# -*- coding: utf-8 -*-

import config
from tpr import *


class GaussianPool(nn.Module):

    def __init__(self, n, tau=None, tau_min=None):
        super(GaussianPool, self).__init__()
        self.n = n  # Number of units in pool
        self.mu = torch.arange(
            n, requires_grad=False,
            dtype=torch.float).to(device=config.device)  # Unit centers
        # self.tau = Parameter(torch.zeros(1))
        if tau is not None:
            self.tau = torch.tensor([tau])
        else:
            # tau ~ Uniform([-.5, +.5])
            self.tau = Parameter(torch.rand(1) - 0.5)
            #self.tau = Parameter(-2.0 * torch.ones(1))
        if tau_min is not None:
            self.tau_min = tau_min
        else:
            self.tau_min = 0.0
        # self.eps = torch.tensor([np.log(1.0e-7),], requires_grad=False)

    def forward(self, posn, mask=None):
        """
        Map batch of soft positions to attention distributons 
        over discrete positions (output is nbatch x drole)
            posn [nbatch x 1], mu [n] => attn [nbatch x n]
        optionally apply log-domain mask [nbatch x n]
        """
        mu = self.mu  # apply relu6 to tau?
        # print(self.tau.item())
        tau = exp(self.tau) + self.tau_min
        # hardtanh(self.tau, 0.0, 10.0)
        # if self.tau_min is not None:
        #    tau = tau + self.tau_min
        attn = -tau * (posn - mu)**2.0  # distance between posn and centers
        if mask is not None:  # apply log-domain mask
            attn = attn + mask

        try:
            assert not np.any(np.isnan(attn.cpu().data.numpy()))
        except:
            AssertException
        if np.any(np.isnan(attn.cpu().data.numpy())):
            print(f"(1) attention value is nan {str(attn.data.numpy())}")
            print(f"mask: {mask}")
            print(f"tau: {tau}")
            print(f"log tau: {self.tau.item()}")
            print(f"posn: {posn}")
            print(f"mu: {mu}")
            sys.exit(0)

        attn = softmax(attn, dim=1)  # Normalize
        # attn = gumbel_softmax(attn, hard = False, dim = 1)
        assert not np.any(np.isnan(attn.cpu().data.numpy())), (
            f"(2) attention value is nan {str(attn.data.numpy())} "
            f"from {posn.data.numpy()}")
        assert np.all(attn.cpu().data.numpy() <= 1.0
                     ), f"(3) attention value greater than 1"

        self.trace = {'tau': tau.clone().detach()}
        return attn


class GaussianPool2D(nn.Module):
    """
    Smooth probability distribution over ordinal positions with
    ~ Gaussian convolution [xxx can this be implemented directly?]
    """

    def __init__(self, n, tau=None):
        super(GaussianPool2D, self).__init__()
        self.n = n  # number of ordinal positions
        # neg squared distance matrix on ordinal positions
        mu = torch.arange(n).type(torch.FloatTensor)
        self.M = mu.view(n, 1) - mu.view(1, n)
        self.M = -torch.pow(self.M, 2.0)
        if tau:  # attentional precision
            self.tau = Parameter(torch.randn(1) * tau)
        else:
            self.tau = Parameter(torch.randn(1))

    def forward(self, b):
        n, M, tau = self.n, self.M, relu6(self.tau)
        dist = exp(log_softmax(tau * M, 0))
        beta = b.mm(dist)
        beta = normalize(beta, 1, 1)
        return beta


# xxx deprecated; no longer used
def posn2attn(posn, tau, n=None):
    """
    Map soft position to attention distribution over discrete positions
    """
    mu = torch.arange(0, n).type(torch.FloatTensor)
    attn = posn - mu
    attn = -tau * torch.pow(attn, 2.0)  # rbf scores
    attn = log_softmax(attn, 0)  # log_softmax
    attn = exp(attn)  # convert to prob distrib
    return attn


def test():
    n, nbatch = 10, 1
    tau_min = 2.0
    config.device = "cpu"
    config.tau_min = 0.0
    pool = GaussianPool(n, tau_min=tau_min)
    for i in range(n):
        x = float(i) * torch.ones(nbatch, 1)
        attn = pool(x)
        print(np.round(attn.data[0].numpy(), 2))

    # gp2 = GaussianPool2D(n, tau)
    # print(gp2.M)
    # b = torch.FloatTensor(nbatch,n)
    # b.data[:] = 0.0; b.data[0,0] = 1.0; b.data[0,4] = 0.1; b.data[1,9] = 1.0
    # print(gp2(b))


if __name__ == "__main__":
    test()
