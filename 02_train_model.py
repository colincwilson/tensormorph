# -*- coding: utf-8 -*-

import configargparse, sys
from pathlib import Path
# Specify local directory containing modules and
# import __all__ modules as specified in __init__.py
sys.path.append(str(Path.cwd() / '../phon'))
sys.path.append(str(Path.cwd() / 'tensormorph'))
from tensormorph import *

# Train and test model using preprocessed data set
# (see notes on data format), thin wrapper for tensormorph
# Examples:
# python 00train_model.py # chamorro -um- infixation
# python 00train_model.py --data hungarian/hungarian_dat_sg --features one_hot
# python 00train_model.py --data english/english_un --features one_hot
# python 00train_model.py --data Marcus1999/marcus_aba --features one_hot --reduplication
# python 00train_model.py --data synth_redup/data/partial_and_initial_cv_100 --features one_hot --reduplication
# python 00train_model.py --data unimorph/que_noun --morphosyn unimorph

parser = configargparse.ArgParser(
    config_file_parser_class=configargparse.YAMLConfigFileParser)
parser.add(
    '--data_dir',
    type=str,
    default='../tensormorph_data',
    help='directory containing data')
parser.add(
    '--data',
    type=str,
    default='chamorro/chamorro_um',
    help='path to pickled dataset (str)')
parser.add(
    '--features',
    type=str,
    choices=['hayes_features.csv', 'panphon_ipa_bases.csv', \
             'riggle_features.csv','one_hot'],
    default='hayes_features.csv',
    help='feature file (str)')
parser.add(
    '--morphosyn',
    type=str,
    choices=['default', 'unimorph'],
    default='default',
    help='morphosyntactic embedder')
parser.add(
    '--phonology',  # xxx rename
    type=int,
    default=0,
    help='size of phonological constraint bank')
parser.add(
    '--dmorphophon',
    type=int,
    default=0,
    help='size of morphophonological context')
parser.add(
    '--reduplication',
    action='store_true',
    help='use reduplication cogrammar (bool)')
parser.add('--batch_size', type=int, default=12)
parser.add('--max_epochs', type=int, default=20)
parser.add(
    '--config',
    default='model_config.yaml',
    help='path to configuration file (str)')
args = parser.parse_args()
print(args)
#dataset = args.data
# xxx save args as pkl for model loading after run

tensormorph.init(args)
tensormorph.train_and_evaluate()
#tensormorph.sample_analysis()

# # # # # # # # # #
# Playground
if 0:
    import sequence_generator
    sequence_generator.train()
