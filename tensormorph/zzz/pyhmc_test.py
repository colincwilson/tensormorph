import numpy as np
import torch
from torch import nn
from pyhmc import hmc

# define your probability distribution
class Distrib(nn.Module):
    def __init__(self, n):
        super(Distrib, self).__init__()
        self.x = nn.Parameter(torch.randn(n))
    
    def forward(self, x, ivar):
        self.x.data = torch.tensor(x)
        ivar = torch.tensor(ivar)
        logp = -0.5 * torch.sum(ivar * self.x**2, -1)
        return logp
    
    def logprob(self, x, ivar):
        self.zero_grad()
        logp = self(x, ivar)
        logp.backward()
        grad = self.x.grad
        logp = np.array(logp.data.numpy(), dtype=np.double)
        grad = np.array(grad.data.numpy(), dtype=np.double)
        print(logp)
        return logp, grad

#def logprob(x, ivar):
#    logp = -0.5 * np.sum(ivar * x**2)
#    grad = -ivar * x
#    print(logp, grad)
#    return logp, grad
# run the sampler
n = 5
p = Distrib(n)
x0 = 10.0 * np.ones(n) # np.random.randn(n)
ivar = 1. / np.random.rand(n)
samples = hmc(p.logprob, x0=x0, args=(ivar,), n_samples=100,
    display=0)
print(samples)
# Optionally, plot the results (requires an external package)
#import triangle  # pip install triangle_plot
#figure = triangle.corner(samples)
#figure.savefig('triangle.png')