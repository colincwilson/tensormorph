import torch
from tpr import *
from scanner import *
import config


def run():
    # All-prefix-sum and all-suffix-sum
    X = torch.abs(torch.randn((1, 2, config.nrole)))
    Xprefix = scan(X, 'LR->', 0)  # prefix, exclusive
    Xsuffix = scan(X, '<-RL', 0)  # suffix, exclusive
    print(X[0])
    print(Xprefix[0])
    print(Xsuffix[0])

    # Directional inhibition
    # xxx fixme
    #DirectionalInhibition.test()