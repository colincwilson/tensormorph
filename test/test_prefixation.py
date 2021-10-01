# -*- coding: utf-8 -*-

# Add a prefix to a stem

import sys
import matplotlib.pyplot as plt
sys.path.append('../src')
from environ import config
from tpr import *
from combiner import Combiner

import test_environ
config.recorder = Recorder()
combiner = Combiner()
combiner.morph_attender.tau = Parameter(2.0 * torch.ones(1))
combiner.posn_attender.tau = Parameter(2.0 * torch.ones(1))

# Single stem as batch, pivot after initial boundary
nbatch = 1
stem = config.form_embedder.string2tpr('p a k').unsqueeze(0)
copy_stem = hardtanh(stem[:,0,:], 0, 1)
pivot = torch.zeros(1, config.nrole)
pivot[:,0] = 1.0

# Single affix as batch
affix = config.form_embedder.string2tpr('r e', delim=False).unsqueeze(0)
unpivot = 1.0 - hardtanh(affix[:,0,:], 0, 1)
copy_affix = hardtanh(affix[:,0,:], 0, 1)

# Output
output = combiner(stem, affix,
                copy_stem, copy_affix,
                pivot, unpivot,
                config.drole)
print (stem.narrow(1,0,5).data.numpy())
print (np.round(output.narrow(1,0,5).data.numpy(), 3))

# Print traces
record = config.recorder.dump()
print (record['combiner-morph_indx'][0,:].t())
print (record['combiner-stem_indx'][0,:].t())
print (record['combiner-affix_indx'][0,:].t())
print (record['combiner-output_indx'][0,:].t())

x = record['combiner-morph_indx'][0,:].t().data.numpy()
y = record['combiner-stem_indx'][0,:].t().data.numpy()
plt.plot(x,y); plt.show()