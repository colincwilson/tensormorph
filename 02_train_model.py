# -*- coding: utf-8 -*-

import argparse, configargparse, sys
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

# Commandline arguments / yaml file specifying data and model specs
parser = configargparse.ArgParser(
    default_config_files=['model_config.yaml'],
    config_file_parser_class=configargparse.YAMLConfigFileParser)

# Custom config file
parser.add(
    '--config',
    type=str,
    is_config_file=True,
    help='path to configuration file (str)')

parser.add('--data_dir', type=str, help='directory containing data (str)')
parser.add('--data_pkl', type=str, help='path to pickled dataset (str)')
parser.add(
    '--features',
    type=str,
    choices=['hayes_features.csv', 'panphon_ipa_bases.csv', \
             'riggle_features.csv','one_hot'],
    default='hayes_features.csv',
    help='feature file (str)')
parser.add_argument(
    '--morphology',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='apply morphology (deactivate with --no-morphology)')
parser.add(
    '--morphosyn',
    type=str,
    choices=['default', 'unimorph'],
    default='default',
    help='morphosyntactic embedder')
parser.add(
    '--dmorphophon',
    type=int,
    default=0,
    help='size of morphophonological context')
parser.add(
    '--phonology',  # xxx rename
    type=int,
    default=0,
    help='size of phonological constraint bank')
parser.add(
    '--reduplication',
    action='store_true',
    help='use reduplication cogrammar (bool)')
parser.add('--batch_size', type=int, default=12)
parser.add('--max_epochs', type=int, default=20)

# Config file / commandline options
args, _ = parser.parse_known_args()

# Default simulation (use if no data specified)
if args.data_pkl is None:
    args.data_dir = '../tensormorph_data/chamorro'
    args.data_pkl = 'chamorro_um.pkl'

# Default data directory (relative to config)
if args.data_dir is None:
    args.data_dir = Path(args.config).parent

print(args)
# xxx save args as pkl for model loading after run

tensormorph.init(args)
tensormorph.train_and_evaluate()
#tensormorph.sample_analysis()

# # # # # # # # # #
# Playground
if 0:
    import sequence_generator
    sequence_generator.train()
