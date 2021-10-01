# -*- coding: utf-8 -*-

# Copy all symbols of a stem except the 
# initial word boundary; illustrates action 
# of combiner with minimal stem markup and 
# null affix

import sys
sys.path.append('../src')
from environ import config
from tpr import *
from combiner import Combiner

import test_environ
combiner = Combiner()
combiner.morph_attender.tau = Parameter(4.0 * torch.ones(1))
combiner.posn_attender.tau = Parameter(4.0 * torch.ones(1))

# Single stem as batch, no pivoting, 
# copy stem symbols except initial boundary
nbatch = 1
stem = config.form_embedder.string2tpr('p a k u').unsqueeze(0)
pivot = torch.zeros((nbatch, config.nrole))
copy_stem = torch.ones((nbatch, 1, config.nrole)) * \
                hardtanh(torch.narrow(stem,1,0,1)) *\
                (1.0 - hardtanh(torch.narrow(stem,1,1,1)))
                #(1.0 - hardtanh(torch.narrow(stem,1,2,1)))
copy_stem = copy_stem.squeeze(1)

# Null affix
affix = torch.zeros((nbatch, config.dfill, config.nrole))
unpivot = torch.zeros((nbatch, config.nrole))
copy_affix = torch.zeros((nbatch, config.nrole))

# Output
output = combiner(stem, affix,
                copy_stem, copy_affix,
                pivot, unpivot, config.drole)
print (stem.narrow(1,0,5).data.numpy())
print (np.round(output.narrow(1,0,5).data.numpy(), 3))