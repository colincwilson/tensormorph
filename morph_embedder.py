#!/usr/bin/env python
# -*- coding: utf-8 -*-
# todo
# - apply dimensionality reduction to observed morphological embeddings
# - allow morph embeddings to be trained

import tpr
from tpr import *
import unimorph_util
from unimorph_util import tag2keyvalue
import pandas as pd
from functools import reduce
from collections import Counter
#from sklearn.decomposition import PCA
import re, sys

one = torch.ones(1)

class MorphEmbedder():
    @staticmethod
    def get_embedder(language=None, dat=None):
        if language is None and dat is not None:
            return UnimorphEmbedder(dat)
        if language == 'hebrew':
            return HebrewEmbedder()
        if language == 'hindi':
            return HindiEmbedder()
        if language is None and dat is None:
            return DummyEmbedder()


class UnimorphEmbedder(MorphEmbedder):
    def __init__(self, dat):
        print('collecting dimensions and labels from morphological tags ...')
        # convert all morph tags to unimorph dimension=label lists
        dat['morph'] = [x.lower() for x in dat['morph']]
        tags = [tag2keyvalue(x) for x in dat['morph']]
        tags_unk = [x for x in tags if re.search('[?]',x)]
        if len(tags_unk)>0:
            print('tags with unknown dimensions or labels:', tags_unk)
        #print unimorph_util.labels_without_dimensions

        # collect dimension=label types
        dimlabs = [set(x.split(';')) for x in tags]
        dimlabs = reduce((lambda x,y: x | y), dimlabs, set([]))
        dimlabs = [x.split('=') for x in dimlabs]
        #print dimlabs

        # collect labels within each dimension
        dims = list(set([x[0] for x in dimlabs])); dims.sort()
        dim2labels = {}
        for dim in dims:
            dim2labels[dim] = [x[1] for x in dimlabs if x[0]==dim]
            dim2labels[dim].sort()
        label2dim = {y:x for x in dim2labels for y in dim2labels[x]}
        print('dimensions:', dims)
        print('dimension => labels:', dim2labels)
        print('label => dimension:', label2dim)

        # make embedding map for each dimension
        dim2embed = {}
        dmorph = 0
        for dim in dims:
            n = len(dim2labels[dim])
            dmorph += n
            dim2embed[dim] = torch.cat([\
                torch.eye(n),\
                torch.zeros(n,1)\
                ], 1)
        print('dimensionality of morphological embedding:', dmorph)

        self.dims = dims
        self.dim2labels = dim2labels
        self.label2dim = label2dim
        self.dim2embed = dim2embed
        self.dmorph = dmorph+1
        self.pca = None

    def embed(self, morph):
        dims, dim2labels, dim2embed = self.dims, self.dim2labels, self.dim2embed
        tags = tag2keyvalue(morph).split(';')
        tags = [x.split('=') for x in tags]
        tags = {x[0]:x[1] for x in tags}
        embeds = [one,]
        for dim in dims:
            lab = tags[dim] if dim in tags else None
            indx = -1 if lab is None else dim2labels[dim].index(lab)
            embeds.append(dim2embed[dim][:,indx])
        embed = torch.cat(embeds, 0)
        if self.pca is not None:
            embed = self.pca.transform(embed.unsqueeze(0).data.numpy())
            embed = torch.FloatTensor(embed).squeeze(0)
            return embed
        return embed

    def reduce_dimension(self, dat):
        embed = self.embed
        # collect unique tags from data, embed each one,
        # perform dimensionality reduction on the result
        tag_types = Counter([x for x in dat.morph])
        print(tag_types)
        for tag in tag_types:
            print(tag, '->', tag_types[tag])
        sys.exit(0)
        tags = list(set([x for x in dat.morph]))
        print('number of unique morphological tags:', len(tags))
        embeds = [embed(x).unsqueeze(-1) for x in tags]
        M = torch.cat(embeds, 1).t()
        M = M.data.numpy()
        print('input matrix dimensionality:', M.shape)
        pca = PCA(); pca.fit(M)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        ndims = len([x for x in cumvar if x<0.999])
        print('reducing to dimensionality to:', ndims)
        self.pca = PCA(n_components=ndims); self.pca.fit(M)
        self.dmorph = ndims


class HebrewEmbedder(MorphEmbedder):
    def __init__(self):
        dim2labels = {\
            'tense': ['BEINONI', 'FUTURE', 'IMPERATIVE', 'INFINITIVE', 'PAST', 'PRESENT'],\
            'person': ['E', 'FIRST', 'SECOND', 'THIRD'],\
            'gender': ['E', 'F', 'M', 'MF'],\
            'number': ['E', 'PLURAL', 'SINGULAR'],\
            'complete': ['COMPLETE', 'MISSING']\
            }
        dims = ['tense', 'person', 'gender', 'number', 'complete']
        label2dim = {y:x for x in dim2labels for y in dim2labels[x]}
        dim2embed = {}
        dmorph = 0
        for dim in dims:
            n = len(dim2labels[dim])
            dmorph += n
            dim2embed[dim] = torch.cat([\
                torch.eye(n)
                ], 1)
        print('dimensionality of morphological embedding:', dmorph)

        self.dims = dims
        self.dim2labels = dim2labels
        self.label2dim = label2dim
        self.dim2embed = dim2embed
        self.dmorph = dmorph+1

    def embed(self, morph):
        dims, dim2labels, dim2embed = self.dims, self.dim2labels, self.dim2embed
        tags = zip(dims, morph.split('+'))
        tags = [(x, re.sub('.*=', '', y)) for (x,y) in tags]
        tags = {x[0]:x[1] for x in tags}
        embeds = [one,]
        for dim in dims:
            lab = tags[dim] if dim in tags else None
            indx = 999 if lab is None else dim2labels[dim].index(lab)
            embeds.append(dim2embed[dim][:,indx])
        return torch.cat(embeds, 0)


class HindiEmbedder(MorphEmbedder):
    def __init__(self):
        dim2labels = {\
            'case': ['direct', 'oblique', 'vocative'],\
            'number': ['singular', 'plural'],\
            'gender': ['masculine', 'feminine', 'neuter']\
        }
        dims = ['case', 'number', 'gender']
        label2dim = {y:x for x in dim2labels for y in dim2labels[x]}
        dim2embed = {}
        dmorph = 0
        for dim in dims:
            n = len(dim2labels[dim])
            dmorph += n
            dim2embed[dim] = torch.cat([\
                torch.eye(n)
                ], 1)
        print('dimensionality of morphological embedding:', dmorph)

        self.dims = dims
        self.dim2labels = dim2labels
        self.label2dim = label2dim
        self.dim2embed = dim2embed
        self.dmorph = dmorph+1

    def embed(self, morph):
        dims, dim2labels, dim2embed = self.dims, self.dim2labels, self.dim2embed
        tags = zip(dims, morph.split(':'))
        tags = [(x, re.sub('.*=', '', y)) for (x,y) in tags]
        tags = {x[0]:x[1] for x in tags}
        embeds = [one,]
        for dim in dims:
            lab = tags[dim] if dim in tags else None
            indx = -1 if lab is None else dim2labels[dim].index(lab)
            embeds.append(dim2embed[dim][:,indx])
        return torch.cat(embeds, 0)


class DummyEmbedder(MorphEmbedder):
    def __init__(self):
        self.dmorph = 1

    def embed(self, morph):
        return one


# unit test
if False:
    datfile = '/Users/colin/Dropbox/TensorProductStringToStringMapping/conll-sigmorphon2018/task1/all/romanian-train-high'
    dat = pd.read_table(datfile, header=None)
    dat.columns = ['stem', 'output', 'morph']
    MorphEmbedder = MorphEmbedder.get_embedder(None, dat)
