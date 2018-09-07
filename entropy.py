#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr
from tpr import *

# entropy of binomial variable(s) p
def binary_entropy(p):
    p = hardtanh(p, .001, .999)
    h = -(p * torch.log(p) + (1.0-p) * torch.log(1.0-p))
    return h