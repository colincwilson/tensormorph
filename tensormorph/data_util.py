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
        idx, source, target, morphosyn = \
            ex['idx'], ex['source'], ex['target'], ex['morphosyn']
        source_len, target_len = \
            ex['source_len'], ex['target_len']
        form_embedder = config.form_embedder
        morphosyn_embedder = config.morphosyn_embedder

        try:
            source_str = str_util.add_delim(source)
            source = form_embedder.string2tpr(source_str, delim=False)
        except Exception as e:
            print(f'Error embedding source {source_str}')
            print(e)
            sys.exit(0)

        try:
            target_str = str_util.add_delim(target)
            target = form_embedder.string2tpr(target_str, delim=False)
            target_id, _ = form_embedder.string2idvec(
                target_str, delim=False, pad=True)
        except Exception as e:
            print(f'Error embedding target {target_str}')
            print(e)
            sys.exit(0)

        try:
            morphosyn_str = morphosyn
            morphosyn = morphosyn_embedder.embed(morphosyn_str)
        except Exception as e:
            print(f'Error embedding morphosyn {morphsyn_str}')
            print(e)
            sys.exit(0)

        dict.__init__(
            self,
            {
                'idx': idx,
                'source': source,
                'source_str': source_str,
                'source_len': source_len,
                'target': target,
                'target_id': target_id,
                'target_str': target_str,
                'target_len': target_len,
                'morphosyn': morphosyn,
                'morphosyn_str': morphosyn_str,
                #'morphospec': morphospec,
            })


def morph_batcher(batch):
    """
    collate_fn for list of MorphDatapoints
    """
    idx = torch.LongTensor([ex['idx'] for ex in batch])
    source = torch.stack([ex['source'] for ex in batch], 0)
    target_id = torch.stack([ex['target_id'] for ex in batch], 0)
    target = torch.stack([ex['target'] for ex in batch], 0)
    morphosyn = torch.stack([ex['morphosyn'] for ex in batch], 0)
    source_str = [ex['source_str'] for ex in batch]
    target_str = [ex['target_str'] for ex in batch]
    morphosyn_str = [ex['morphosyn_str'] for ex in batch]
    source_len = torch.LongTensor([ex['source_len'] + 2 for ex in batch])
    target_len = torch.LongTensor([ex['target_len'] + 2 for ex in batch])
    max_len = np.max(source_len.numpy() + target_len.numpy())
    batch = {
        'idx':
            idx,
        'source_str':
            source_str,
        'target_str':
            target_str,
        'morphosyn_str':
            morphosyn_str,
        'source':
            Morph(form=source, form_str=source_str, length=source_len),
        'target':
            Morph(
                form=target,
                form_id=target_id,
                form_str=target_str,
                length=target_len),
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
                source_regex=None,
                target_regex=None,
                morphosyn_regex=None):
    """
    Extract data subset defined by regexes on source / target / morphosyn
    (applied conjunctively to raw data prior to embedding)
    """
    dat1 = dataset['dat']
    if source_regex is not None:
        dat1 = dat1[dat1.source.str.match(source_regex)].\
                reset_index(drop = True)
        #print(stem_regex, len(dat1))
    if target_regex is not None:
        dat1 = dat1[dat1.target.str.match(target_regex)].\
                reset_index(drop = True)
        #print(output_regex, len(dat1))
    if morphosyn_regex is not None:
        dat1 = dat1[dat1.morphosyn.str.match(morphosyn_regex)].\
                reset_index(drop = True)

    return dat1
