# -*- coding: utf-8 -*-

import collections, sys
import numpy as np
import pandas as pd
import torch

import config
from phon import str_util
from morph import Morph


class MorphDataset(torch.utils.data.Dataset):
    """
    Container for train | val | test data embeddings
    """

    def __init__(self, dat):
        self.dat_embed = self.embed(dat)
        self.ndata = len(dat)

    def __getitem__(self, indx):
        return self.dat_embed[indx]

    def __len__(self):
        return self.ndata

    def embed(self, dat):
        return [MorphDatapoint(ex) \
            for i, ex in dat.iterrows()]


class MorphDatapoint(dict):
    """
    Embedding of a single example
    """

    def __init__(self, ex):
        idx, stem, output, morphosyn = \
            ex['idx'], ex['stem'], ex['output'], ex['morphosyn']
        stem_len, output_len = \
            ex['stem_len'], ex['output_len']
        form_embedder = config.form_embedder
        morphosyn_embedder = config.morphosyn_embedder

        try:
            stem_str = str_util.add_delim(stem)
            stem = form_embedder.string2tpr(stem_str, delim=False)
        except Exception as e:
            print(f'error embedding stem {stem_str}')
            print(e)
            sys.exit(0)

        try:
            output_str = str_util.add_delim(output)
            output = form_embedder.string2tpr(output_str, delim=False)
            output_id, _ = form_embedder.string2idvec(
                output_str, delim=False, pad=True)
        except Exception as e:
            print(f'error embedding output {output_str}')
            print(e)
            sys.exit(0)

        try:
            morphosyn_str = morphosyn
            morphosyn = morphosyn_embedder.embed(morphosyn_str)
        except Exception as e:
            print(f'error embedding morphosyn {morphsyn_str}')
            print(e)
            sys.exit(0)

        dict.__init__(
            self,
            {
                'idx': idx,
                'stem': stem,
                'stem_str': stem_str,
                'stem_len': stem_len,
                'output': output,
                'output_id': output_id,
                'output_str': output_str,
                'output_len': output_len,
                'morphosyn': morphosyn,
                'morphosyn_str': morphosyn_str,
                #'morphospec': morphospec,
            })


def morph_batcher(batch):
    """
    collate_fn for list of MorphDatapoints
    """
    idx = torch.LongTensor([ex['idx'] for ex in batch])
    stem = torch.stack([ex['stem'] for ex in batch], 0)
    output_id = torch.stack([ex['output_id'] for ex in batch], 0)
    output = torch.stack([ex['output'] for ex in batch], 0)
    morphosyn = [
        torch.stack([ex['morphosyn'][i]
                     for ex in batch], 0)
        for i in range(config.ndim)
    ]
    #morphospec = torch.stack(
    #    [ex['morphospec'] for ex in batch], 0)
    stem_str = [ex['stem_str'] for ex in batch]
    output_str = [ex['output_str'] for ex in batch]
    morphosyn_str = [ex['morphosyn_str'] for ex in batch]
    stem_len = torch.LongTensor([ex['stem_len'] + 2 for ex in batch])
    output_len = torch.LongTensor([ex['output_len'] + 2 for ex in batch])
    max_len = np.max(stem_len.numpy() + output_len.numpy())
    batch = {
        'idx':
            idx,
        'stem_str':
            stem_str,
        'output_str':
            output_str,
        'morphosyn_str':
            morphosyn_str,
        'stem':
            Morph(form=stem, form_str=stem_str, length=stem_len),
        'output':
            Morph(
                form=output,
                form_id=output_id,
                form_str=output_str,
                length=output_len),
        'morphosyn':
            morphosyn,
        #'morphospec': morphospec,
        'pred':
            None,
        'max_len':
            max_len
    }
    return batch


# # # # # Deprecated # # # # #


def subset_data(dataset,
                stem_regex=None,
                output_regex=None,
                morphosyn_regex=None):
    """
    Extract data subset defined by regexes on stem / output / morphosyn
    (applied conjunctively to raw data prior to embedding)
    """
    dat1 = dataset['dat']
    if stem_regex is not None:
        dat1 = dat1[dat1.stem.str.match(stem_regex)].\
                reset_index(drop = True)
        #print(stem_regex, len(dat1))
    if output_regex is not None:
        dat1 = dat1[dat1.output.str.match(output_regex)].\
                reset_index(drop = True)
        #print(output_regex, len(dat1))
    if morphosyn_regex is not None:
        dat1 = dat1[dat1.morphosyn.str.match(morphosyn_regex)].\
                reset_index(drop = True)

    return dat1
