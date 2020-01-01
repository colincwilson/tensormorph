#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
import numpy as np
import pandas as pd
import torch, sys

class Recorder:
    def __init__(self):
        self.record = {}

    def set_values(self, prefix, keyvals):
        # Record (key,value) pairs
        for x,y in keyvals.items():
            self.record[prefix+'-'+x] = y.clone()

    def update_values(self, prefix, keyvals):
        # Update record of (key,value) pairs
        for x,y in keyvals.items():
            x = prefix+'-'+x
            if not x in self.record:
                self.record[x] = [y.clone(),]
            else:
                self.record[x].append(y.clone())
    
    def dump(self, save=False, test=None):
        for x,y in self.record.items():
            if isinstance(y, list):
                y = [yi.unsqueeze(1) for yi in y]
                self.record[x] = torch.cat(y, dim=1)
        if save:
            # Save all recorded objects
            for x,y in self.record.items():
                y = np.clip(y.data.numpy(), -1.0e5, 1.0e5)
                np.save(config.save_dir+'/'+x +'.npy', y)
            # Write filler, role, unbinding matrices
            np.save(config.save_dir+'/filler_matrix.npy', config.F)
            np.save(config.save_dir+'/role_matrix.npy', config.R)
            np.save(config.save_dir+'/unbind_matrix.npy', config.U)
            # Write symbols
            syms = np.array(config.syms)
            np.savetxt(config.save_dir+'/symbols.txt', syms, fmt='%s')
        return self.record

    def init(self):
        self.record = {}