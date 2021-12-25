# -*- coding: utf-8 -*-

# Prepare data for tensormorph

# Examples:
# python 00prepare_data.py --config ../tensormorph_data/chamorro/chamorro_um.yaml
# python 00prepare_data.py --config hungarian/hungarian_dat_sg.yaml
# python 00prepare_data.py --config MarcusEtAl1999/marcus.yaml --test_proportion 0.0
# python 00prepare_data.py --config korean/korean.yaml --dat_file korean_odden --pkl_file korean_odden.pkl --min_train 500

import pickle, re, sys
from pathlib import Path
import configargparse, yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Commandline arguments / yaml fields specifying data specs
parser = configargparse.ArgParser(
    default_config_files=['data_config.yaml'],
    config_file_parser_class=configargparse.YAMLConfigFileParser)

# Custom config file
parser.add(
    '--config',
    type=str,
    is_config_file=True,
    help='path to configuration file (str)')

# Required: Specify data to split or pre-split train and test files
parser.add('--data_file', type=str, help='path to data file (str)')
parser.add('--train_file', type=str, help='path to pre-split train file (str)')
parser.add('--val_file', type=str, help='path to pre-split val file (str)')
parser.add('--test_file', type=str, help='path to pre-split test file (str)')

# Additional options
parser.add('--data_dir', type=str, help='directory containing data (str)')
parser.add('--pkl_file', type=str, help='path to output file (str)')
parser.add('--delim', type=str, help='delimiter in data file (str)')
parser.add('--morphosyn', help='default morphosyntactic specification (str)')
parser.add(
    '--max_len', type=int, help='maximum length of input/output forms (int)')
parser.add(
    '--split_strings',
    action='store_true',
    help='split input/output forms into space-separated (bool)')
parser.add(
    '--lower_strings',
    action='store_true',
    help='lowercase input/output forms (bool)')
parser.add(
    '--substitutions',
    type=yaml.load,
    help='symbol substitutions to apply to input/output forms (dict)')
parser.add(
    '--held_in_stems',
    type=yaml.load,
    nargs='+',
    help='stems necessarily included in the training set (list)')
parser.add(
    '--held_out_stems',
    type=yaml.load,
    nargs='+',
    help='stems necessarily excluded from the training set (list)')
parser.add(
    '--vowels',
    required=True,
    type=yaml.load,
    nargs='+',
    help='vowel symbols (list)')  # xxx no longer obligatory?
parser.add(
    '--val_proportion',
    type=float,
    default=0.10,
    help='approximate proportion of data for validation split (float) ')
parser.add(
    '--test_proportion',
    type=float,
    default=0.40,
    help='approximate proportion of data for test split (float)')
parser.add(
    '--min_train',
    type=int,
    default=0,
    help='enforce minimum size of train split by upsampling (int)')

args = parser.parse_args()
print(args)


def main():
    data_dir = Path(args.data_dir) if args.data_dir is not None \
        else Path(args.config).parent
    delim = ',' if args.delim is None else args.delim
    if args.morphosyn is None or args.morphosyn == 'None':
        colnames = ['stem', 'output', 'morphosyn']
    else:
        colnames = ['stem', 'output']

    if args.data_file is not None:
        # Data to be split into train | val | test
        data_file = data_dir / Path(args.data_file)
        print(f'Preparing data from {data_file} ...')
        data = pd.read_table(
            data_file,
            comment='#',
            sep=delim,
            usecols=colnames,
            names=colnames,
            engine='python')
        split_flag = True
    else:
        # Pre-split train | (val) | test
        # Train
        train_file = data_dir / Path(args.train_file)
        data_train = pd.read_table(
            train_file,
            comment='#',
            sep=delim,
            usecols=colnames,
            names=colnames,
            engine='python')
        data_train['split'] = 'train'

        # Val
        try:
            val_file = data_dir / Path(args.val_file)
            data_val = pd.read_table(
                val_file,
                comment='#',
                sep=delim,
                usecols=colnames,
                names=colnames,
                engine='python')
        except:
            data_val = dat_train.copy()
        data_val['split'] = 'val'

        # Test
        test_file = data_dir / Path(args.test_file)
        data_test = pd.read_table(
            test_file,
            comment='#',
            sep=delim,
            usecols=colnames,
            names=colnames,
            engine='python')
        data_test['split'] = 'test'

        data = pd.concat([data_train, data_val, data_test], 0)
        split_flag = False

    print('Original data')
    data = data.dropna(how='all').reset_index(drop=True)
    data = data.drop_duplicates().reset_index(drop=True)
    data.reset_index()
    max_len = np.max([len(x) for x in data['stem']] +
                     [len(x) for x in data['output']])
    print(data.head())
    print(f'Maximum input/output length: {max_len}')
    print()

    # Default morphosyn
    if args.morphosyn is not None and args.morphosyn != 'None':
        data['morphosyn'] = args.morphosyn

    # Restructure data
    dats = {
        'stem': [x for x in data['stem']],
        'output': [x for x in data['output']],
        'held_in_stems': [],
        'held_out_stems': []
    }

    # Collect segments from stems and outputs prior to string ops
    if not args.split_strings:
        segs = segments(dats, args, sep='')
        print(f'Segments: {segs}')
        print()

    # Apply string ops
    if args.held_in_stems is not None:
        dats['held_in_stems'] = args.held_in_stems

    if args.held_out_stems is not None:
        dats['held_out_stems'] = args.held_out_stems

    if args.split_strings:
        for key, val in dats.items():
            dats[key] = [' '.join(x) for x in val]

    if args.lower_strings:
        for key, val in dats.items():
            dats[key] = [x.lower() for x in val]

    # Apply substitutions
    if args.substitutions is not None:
        print(f'Applying substitutions\n{args.substitutions}')
        for s, r in args.substitutions.items():
            for key, val in dats.items():
                dats[key] = [re.sub(s, r, x) for x in val]
        print()

    data['stem'] = dats['stem']
    data['output'] = dats['output']
    data['stem_len'] = [len(x.split()) for x in dats['stem']]
    data['output_len'] = [len(x.split()) for x in dats['output']]

    # Subset data by max length parameter
    if args.max_len is not None:
        max_len = args.max_len
        data = data[((data.stem_len <= max_len) & (data.output_len <= max_len))]
        data.reset_index()
        #data = data.drop(['stem_len', 'output_len'], axis=1)

    # Index entries
    data['idx'] = range(len(data))

    # Report data
    print(f'Number of unique stem-output pairs: {len(data)}')
    print(data.head())
    print(data.tail())
    print()

    # Collect segments from stems and outputs
    segs = segments(data, args)
    print(f'Segments in the modified data: {segments(data, args)}')
    print()

    # Dataset prior to split
    dataset = {
        'data': data,
        'held_in_stems': dats['held_in_stems'],
        'held_out_stems': dats['held_out_stems'],
        'segments': segs,
        'vowels': args.vowels,
        'max_len': max_len
        #'morphosyn_embedder': morphosyn_embedder
    }

    # Split data
    if split_flag:
        data_train, data_val, data_test = split_data(dataset,
                                                     args.val_proportion,
                                                     args.test_proportion)
        data_train['split'] = 'train'
        data_val['split'] = 'val'
        data_test['split'] = 'test'
    else:
        data_train = data[(data['split'] == 'train')].reset_index()
        data_val = data[(data['split'] == 'val')].reset_index()
        data_test = data[(data['split'] == 'test')].reset_index()

    # Upsample training data to minimum size
    if args.min_train is not None and args.min_train > 0:
        scale = int(float(args.min_train) / len(data_train))
        if scale > 1:
            data_train = pd.concat([data_train] * scale)

    # Dataset with splits
    dataset['data_train'] = data_train
    dataset['data_val'] = data_val
    dataset['data_test'] = data_test
    print(f'train {len(data_train)} | '
          f'val {len(data_val)} | '
          f'test {len(data_test)}')

    # Save full dataset (pkl) and individual splits (csv)
    if args.pkl_file is not None:
        pkl_file = Path(args.pkl_file).with_suffix('.pkl')
        split_file = Path(args.pkl_file).with_suffix('')
    else:
        pkl_file = Path(args.data_file).with_suffix('.pkl')
        split_file = Path(args.data_file).with_suffix('')
    pickle.dump(dataset, open(data_dir / pkl_file, 'wb'))
    data_train.to_csv(
        data_dir / Path(f'{split_file.name}_train.csv'), index=False)
    data_val.to_csv(data_dir / Path(f'{split_file.name}_val.csv'), index=False)
    data_test.to_csv(
        data_dir / Path(f'{split_file.name}_test.csv'), index=False)
    return dataset


def segments(data, args, sep=' '):
    """
    Unique segments in stem or output
    """
    vowels = args.vowels
    segments = set(vowels)
    for stem in data['stem']:
        segments |= set(stem.split(sep)) if sep != '' else set(stem)
    for output in data['output']:
        segments |= set(output.split(sep)) if sep != '' else set(output)
    segments = [x for x in segments]
    segments.sort()
    return (segments)


def split_data(dataset, val_prop, test_prop):
    """
    Random split into train/val/test, handling held_in/out_stems
    """
    data = dataset['data']
    held_in_stems = dataset['held_in_stems']
    held_out_stems = dataset['held_out_stems']

    # Remove held_in and held_out prior to random split
    if held_in_stems is not None:
        held_in = data[(data['stem'].isin(held_in_stems))].\
                    reset_index(drop = True)
        data = data[~(data['stem'].isin(held_in_stems))].\
                    reset_index(drop = True)

    if held_out_stems is not None:
        held_out = data[(data['stem'].isin(held_out_stems))].\
                    reset_index(drop = True)
        data = data[~(data['stem'].isin(held_out_stems))].\
                    reset_index(drop = True)

    # Split non-held into train and val+test
    nontrain_size = (val_prop + test_prop)
    data_train, data_nontrain = \
        train_test_split(data, test_size = nontrain_size)

    # Split rest into val and test
    test_size = test_prop / (val_prop + test_prop)
    data_val, data_test = \
        train_test_split(data_nontrain, test_size = test_size)

    # Combine held_in and held_out with splits
    if held_in_stems is not None:
        data_train = pd.concat([held_in, data_train]).\
                    reset_index(drop = True)

    if held_out_stems is not None:
        data_test = pd.concat([held_out, data_test]).\
                    reset_index(drop = True)

    splits = [('train', data_train), ('val', data_val), ('test', data_test)]
    for (split, data_split) in splits:
        print(f'{split} split')
        print(data_split.head())
        print()

    return data_train, data_val, data_test


if __name__ == '__main__':
    main()
