#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
import tpr
from tpr import *
from affixer import Affixer

class Model(nn.Module):
    def __init__(self, node='root'):
        super(Model, self).__init__()
        self.affixer = Affixer()

    def forward(self, stem, morph, max_len):
        return self.affixer(stem, morph, max_len)
