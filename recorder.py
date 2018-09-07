#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr
import torch
import sys

class Recorder:
    def __init__(self):
        self.record = {}

    # record (key,value) pairs
    def set_recording(self, vals):
        for x,y in vals.iteritems():
           self.record[x] = y.clone()

    # update recording (key,value) pairs
    def update_recording(self, vals):
        for x,y in vals.iteritems():
            if not x in self.record:
                self.record[x] = [y.clone(),]
            else:
                self.record[x].append(y.clone())
    
    def dump(self):
        for x,y in self.record.iteritems():
            if isinstance(y, list):
                y = [yi.unsqueeze(1) for yi in y]
                self.record[x] = torch.cat(y, dim=1)
        return self.record

    def init(self):
        self.record = {}