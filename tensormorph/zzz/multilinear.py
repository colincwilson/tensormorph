# -*- coding: utf-8 -*-

import string
import numpy as np
import config
from tpr import *


class Multilinear(nn.Module):
    """
    Multilinear layer (outer product -> bottleneck -> output)
    adapted from:
    https://github.com/santient/pytorch-multilinear/blob/main/multilinear.py
    """

    def __init__(self, in_features, hidden_features, out_features, bias=True):
        super(Multilinear, self).__init__()
        #print(in_features)
        self.in_features = tuple(in_features)
        prod_features = np.prod(in_features)
        #print(prod_features, hidden_features, out_features, bias)
        self.prd2hid = nn.Linear(prod_features, hidden_features, bias=bias)
        self.hid2out = nn.Linear(hidden_features, out_features, bias=bias)
        dimchars = string.ascii_lowercase[0:len(self.in_features)]
        self.einsum_str = ','.join([f'z{x}' for x in dimchars]) + \
                          '->' + 'z' + ''.join(dimchars)
        #print(self.einsum_str)

    def forward(self, *inputs):
        prd = einsum(self.einsum_str, *inputs)
        prd = prd.flatten(start_dim=1)
        hid = self.prd2hid(prd)
        out = self.hid2out(hid)
        return out


def main():
    M = Multilinear((2, 2, 2), 3, 3)
    x1 = torch.rand(3, 2)
    x2 = torch.rand(3, 2)
    x3 = torch.rand(3, 2)
    val = M((x1, x2, x3))
    print(val)


if __name__ == "__main__":
    main()