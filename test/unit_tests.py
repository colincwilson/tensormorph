# -*- coding: utf-8 -*-

import sys
sys.path.append('./src')
from environ import config
from tpr import *
from correspondence import Correspondence

# # # # # # # # #
# Correspondence indices and faithful updates
config.nrole = 8
config.dfill = 3
correspondence = Correspondence()
X = torch.rand(2, config.dfill, config.nrole)
Y = correspondence.indexify(X)
Y = Y[:,:,0:4]
Y = torch.cat([Y, Y[:,:,1:]], -1)
print (Y.shape)
print (Y)

print ('faithful updates')
faith, alpha = correspondence(Y, 8)
print (np.round(faith.data.numpy(), 3))
print (np.round(alpha.data.numpy(), 3))
sys.exit(0)


# # # # # # # # #
# Self-attention from correspondence indices
indices0 = torch.linspace(
    1.0, config.nrole, config.nrole, requires_grad=False)
indices0[-1] = 0.0
indices0 = torch.cat([indices0, indices0], 0)
indices0 = indices0.unsqueeze(0).unsqueeze(0)
indices1 = torch.linspace(
    1.0, config.nrole, config.nrole, requires_grad=False)
indices1 = torch.cat([indices1, indices1], 0)
indices1 = indices1.unsqueeze(0).unsqueeze(0)
indices = torch.cat([indices0, indices1], 0)
indices = indices.transpose(2,1)
print (indices.shape)
print (indices)

#dist = F.unsqueeze(0) - f.unsqueeze(-1)
#dist = (dist**2.0).sum(1)
diffs = indices.unsqueeze(2) - indices.unsqueeze(1)
diffs = torch.sum(diffs**2.0, -1)
sims = -diffs
print (sims.shape)
print (sims[0,:,:])
print ()

# correspondence applies LR-> (tril) or <-RL (triu)
sims = torch.tril(sims)
print (sims[0,:,:])