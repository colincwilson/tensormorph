#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr
from affixer import Affixer
from decoder import Decoder, LocalistDecoder

class Model():
    def __init__(self):
        affixer = Affixer(); affixer.init()
        decoder = Decoder() if 0 else LocalistDecoder()
        tpr.decoder = decoder
        self.affixer = affixer
        self.decoder = decoder
