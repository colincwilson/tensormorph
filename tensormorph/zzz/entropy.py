# -*- coding: utf-8 -*-

from tpr import *

def binary_entropy(p):
    """
    Entropy of binomial variables p
    """
    p = hardtanh(p, .001, .999)
    h = -(p * torch.log(p) + (1.0-p) * torch.log(1.0-p))
    return h