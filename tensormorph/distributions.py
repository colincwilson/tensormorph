# -*- coding: utf-8 -*-

from tpr import *

# xxx add sampling config for softmax, discretize


def rsample(logits, use_softmax=1, discretize=0):
    """
    Draw batch of samples from OneHotCategorical 
    and add log_prob to global loss -or-
    draw batch of reparameterized samples from 
    RelaxedOneHotCategorical, -or- apply softmax 
    with temperature, and optionally discretize
    """
    # Hard sample and log_prob update to loss
    if config.reinforce:
        distrib = OneHotCategorical(logits=logits)
        y_hard = distrib.sample()
        log_prob = distrib.log_prob(y_hard)
        config.log_prob = config.log_prob + log_prob
        return y_hard

    # Softmax or soft reparameterized sample
    temp = config.temperature
    if use_softmax:
        y_soft = softmax(logits / temp, dim=-1)
    else:
        y_soft = RelaxedOneHotCategorical(temperature=temp,
                                          logits=logits).rsample()
    # Discretize
    if discretize or config.discretize:
        # code copied from gumbel_softmax
        # Straight through.
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format).scatter_(
                -1, index, 1.0)
        y_hard.requires_grad = False  # making this explicit
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    #print(logits, '->', ret)
    return ret


def rsample_bin(logits):
    if config.reinforce:
        distrib = Bernoulli(logits=logits)
        y_hard = distrib.sample()
        log_prob = distrib.log_prob(y_hard)
        config.log_prob = config.log_prob + log_prob
        return y_hard
    else:
        sys.exit(0)


class Discrete():
    """
    Discrete distribution over n-dimensional 
    one-hot vectors (uniform if probs is None)
    """

    def __init__(self, probs=None, n=None):
        if probs is None:
            probs = torch.ones(n)
        self.distrib = OneHotCategorical(probs)

    def sample(self, x):
        return self.distrib.sample()

    def log_prob(self, x):
        return self.distrib.log_prob(x)


class SphericalNormal():

    def __init__(self, mu=None, sigma=None, n=None):
        if mu is None and sigma is None:
            mu = torch.zeros(n)
            sigma = torch.eye(n)
        self.distrib = MultivariateNormal(loc=mu, covariance_matrix=sigma)

    def sample(self, x):
        return self.distrib.sample()

    def log_prob(self, x):
        return self.distrib.log_prob(x)


class Geometric():
    """
    Geometric distribution over one-hot vectors
    """

    def __init__(self, p, n):
        #probs = torch.ones(n) * 1.0e-10
        probs = torch.zeros(n)
        for k in range(5):
            probs.data[k] = (1.0 - p)**k * p
        self.distrib = OneHotCategorical(probs=probs)

    def sample(self, x):
        return self.distrib.sample()

    def log_prob(self, x):
        return self.distrib.log_prob(x)


class Exponential():
    """
    Exponential distribution over one-hot vectors
    """

    def __init__(self, alpha, n):
        probs = torch.zeros(n)
        for k in range(5):
            probs.data[k] = np.exp(-k * alpha)
        self.distrib = OneHotCategorical(probs=probs)

    def sample(self, x):
        return self.distrib.sample()

    def log_prob(self, x):
        return self.distrib.log_prob(x)


class OneStep():
    """
    Proposal distribution for form lengths, 
    encoded as one-hot vectors
    """

    def __init__(self, n):
        self.n = n
        self.bernoulli = Bernoulli(torch.tensor([0.5]))

    def sample(self, x):
        n = self.n
        y = torch.zeros_like(x)
        k = np.where(x.data.numpy() == 1)[0][0]
        step = int(self.bernoulli.sample().item())
        k_new = k + (2 * step - 1)
        if k_new < 0:
            k_new = 0
        if k_new >= (n - 1):
            k_new = (n - 1)
        y.data[k_new] = 1.0
        #print(k, step, k_new, x, '->', y); sys.exit(0)
        return y

    def log_prob(self, x):
        return None
