#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re, sys

# pass sys.argv[1:] in as x
def parse(x):
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--nbatch',\
        help='Number of <input,output> pairs in each minibatch')
    argparser.add_argument('--nepoch',\
        help='Number of training epochs')
    args, residue = argparser.parse_known_args(x)
    print ('args:', args)
    print ('residue:', residue)

    if args.nbatch is not None:
        print (args.nbatch)
    return None
