#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch, tpr, sys

class Recorder:
    def __init__(self):
        self.record = {}

    # record (key,value) pairs
    def set_values(self, prefix, keyvals):
        for x,y in keyvals.iteritems():
            self.record[prefix+'-'+x] = y.clone()

    # update record of (key,value) pairs
    def update_values(self, prefix, keyvals):
        for x,y in keyvals.iteritems():
            x = prefix+'-'+x
            if not x in self.record:
                self.record[x] = [y.clone(),]
            else:
                self.record[x].append(y.clone())
    
    def dump(self, save=False, test=None):
        for x,y in self.record.iteritems():
            if isinstance(y, list):
                y = [yi.unsqueeze(1) for yi in y]
                self.record[x] = torch.cat(y, dim=1)
        if save:
            # save all recorded objects
            for x,y in self.record.iteritems():
                y = np.clip(y.data.numpy(), -1.0e5, 1.0e5)
                np.save(tpr.save_dir+'/'+x +'.npy', y)
            # write filler, role, unbinding matrices
            np.save(tpr.save_dir+'/filler_matrix.npy', tpr.F)
            np.save(tpr.save_dir+'/role_matrix.npy', tpr.R)
            np.save(tpr.save_dir+'/unbind_matrix.npy', tpr.U)
            # write symbols
            syms = np.array(tpr.seq_embedder.syms)
            np.savetxt(tpr.save_dir+'/symbols.txt', syms, fmt='%s')
            # write test forms
            if test is not None:
                test.to_csv(tpr.save_dir+'/test.csv', encoding='utf-8')
        return self.record

    def init(self):
        self.record = {}