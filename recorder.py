#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
import numpy as np
import pandas as pd
import torch, sys

class Recorder:
    def __init__(self):
        self.record = {}

    # record (key,value) pairs
    def set_values(self, prefix, keyvals):
        for x,y in keyvals.items():
            self.record[prefix+'-'+x] = y.clone()

    # update record of (key,value) pairs
    def update_values(self, prefix, keyvals):
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
            # save all recorded objects
            for x,y in self.record.items():
                y = np.clip(y.data.numpy(), -1.0e5, 1.0e5)
                np.save(config.save_dir+'/'+x +'.npy', y)
            # write filler, role, unbinding matrices
            np.save(config.save_dir+'/filler_matrix.npy', config.F)
            np.save(config.save_dir+'/role_matrix.npy', config.R)
            np.save(config.save_dir+'/unbind_matrix.npy', config.U)
            # write symbols
            syms = np.array(config.seq_embedder.syms)
            np.savetxt(config.save_dir+'/symbols.txt', syms, fmt='%s')
            # write test forms
            if test is not None:
                test.to_csv(config.save_dir+'/test.csv', encoding='utf-8')
        return self.record

    def init(self):
        self.record = {}