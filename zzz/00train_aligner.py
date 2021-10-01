# -*- coding: utf-8 -*-

# Train alignments
# python 00train_model.py # chamorro -um- infixation
# python 00train_aligner.py --data marcus_aba --features one_hot --reduplication

import sys, argparse
sys.path.append('./src')
from environ import config
import tensormorph, aligner, aligner2
from data_util import morph_batcher

parser = argparse.ArgumentParser(description='Train aligner.')
parser.add_argument('--data', type=str, nargs='?', default='chamorro_um')
parser.add_argument('--features', type=str, nargs='?', 
                    choices=['hayes_features', 'panphon_ipa_bases',
                             'riggle_features', 'one_hot'],
                    default='hayes_features')
parser.add_argument('--reduplication', action='store_true')
# todo: add arguments for morphosyn embeddings
args = parser.parse_args()
print(args)
dataset = args.data

tensormorph.init(
    dataset = dataset,
    features = args.features,
    reduplication = args.reduplication)

batch = morph_batcher(config.train_dat)

aligner = aligner2.Aligner(batch, args.reduplication)
aligner.train()