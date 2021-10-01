# -*- coding: utf-8 -*-

# todo
# - allow morph embeddings to be trained

from tpr import *
from unimorph_util import tag2keyvalue
import pandas as pd
from functools import reduce
from collections import Counter
#from sklearn.decomposition import PCA
import re, sys

verbosity = 10


class MorphosynEmbedder():
    """
    Embed morphosyntactic feature matrices / tags
    """

    @staticmethod
    def get_embedder(language=None, data=None):
        if language is None and data is not None:
            return UnimorphEmbedder(data)
        if language == 'hebrew':
            return HebrewEmbedder()
        if language == 'hindi':
            return HindiEmbedder()
        if language is None and data is None:
            return DummyEmbedder()


class UnimorphEmbedder(MorphosynEmbedder):

    def __init__(self):
        print('Collecting dimensions and labels from morphosyntactic tags ...')
        # Convert observed tags to unimorph dimension=label lists
        tags = set()
        for split in ['data_train', 'data_val', 'data_test']:
            data = config.data[split]
            data['morphosyn'] = [x.lower() for x in data['morphosyn']]
            tags |= set([tag2keyvalue(x) for x in data['morphosyn']])
        #print(unimorph_util.labels_without_dimensions)

        # Collect dimension=label types
        tags = [set(x.split(';')) for x in tags]
        tags = reduce((lambda x, y: x | y), tags, set([]))
        tags_unk = [x for x in tags if re.search('[?]', x)]
        if len(tags_unk) > 0:
            print('Tags with unknown dimensions or labels:', tags_unk)
            sys.exit(0)
        dimlabs = [x.split('=') for x in tags]
        #print(dimlabs)

        # Collect labels within each dimension
        dims = list(set([x[0] for x in dimlabs]))
        dims.sort()
        ndim = len(dims)
        dim2labels = {}
        for dim in dims:
            dim2labels[dim] = [x[1] for x in dimlabs if x[0] == dim]
            dim2labels[dim].sort()
            dim2labels[dim] = ['Ø'] + dim2labels[dim]
        label2dim = {
            y: x for x in dim2labels for y in dim2labels[x] if y != 'Ø'
        }
        dim2size = [len(labels) for dim, labels in dim2labels.items()]
        dmorphosyn = np.sum(dim2size)
        if verbosity > 5:
            print('Summary of dimensions and labels in the data ...')
            print(f'dimensions: {dims}')
            print(f'dimension => labels: {dim2labels}')
            print(f'label => dimension: {label2dim}')
            print(f'dim2size: {dim2size}')
            print(f'dmorphosyn: {dmorphosyn}')

        # Embedding map for each dimension
        dim2embed = {}
        for dim in dims:
            n = len(dim2labels[dim])
            dim2embed[dim] = torch.eye(n, requires_grad=False)
            #dim2embed[dim] = torch.randn(32, n, requires_grad=False)
            #dim2embed[dim] = torch.zeros(dmorphosyn, n, requires_grad=False)
            #dim2embed[dim].data[n_total:(n_total+n),:n] = torch.eye(n)
            #n_total += n
            print(dim)
            print(dim2embed[dim])
        if verbosity > 5:
            print(f'Dimensionality of morphosyntactic embedding: {dmorphosyn}')

        # Matrix mapping each dimensions to its embedding units,
        # and matrix mapping each dimension to its unspecified vector
        Mdim2units = torch.zeros((dmorphosyn, ndim), requires_grad=False)
        Mdim2unspec = torch.zeros((dmorphosyn, ndim), requires_grad=False)
        n_total = 0
        for j, dim in enumerate(dims):
            n = len(dim2labels[dim])
            Mdim2units[n_total:(n_total + n), j] = 1.0
            Mdim2unspec[n_total, j] = 1.0
            n_total += n
        if verbosity > 5:
            print(f'Matrix mapping dimensions to embedding units (transposed)')
            print(Mdim2units.t())
            print(
                f'Matrix mapping dimensions to unspecified vectors (transposed)'
            )
            print(Mdim2unspec.t())

        self.ndim = ndim
        self.dims = dims
        self.dim2size = dim2size
        self.dim2labels = dim2labels
        self.label2dim = label2dim
        self.unit2label = []  # identify each embedding unit
        for dim in dims:
            self.unit2label += [f'{dim}={lab}' for lab in dim2labels[dim]]
        self.dim2embed = dim2embed
        self.Mdim2units = Mdim2units
        self.Mdim2unspec = Mdim2unspec
        self.dmorphosyn = dmorphosyn
        #self.pca = None

    def embed(self, morphosyn):
        """
        Tuple of embeddings of labels in morphosyn
        """
        dims, dim2labels, dim2embed = \
            self.dims, self.dim2labels, self.dim2embed
        ndim = self.ndim
        tags = tag2keyvalue(morphosyn).split(';')
        tags = [x.split('=') for x in tags]
        tags = {x[0]: x[1] for x in tags}
        embeds = []
        #specs = np.zeros(ndim)
        for i, dim in enumerate(dims):
            lab = tags[dim] if dim in tags else 'Ø'
            indx = dim2labels[dim].index(lab)
            embeds.append(dim2embed[dim][:, indx])
            #specs[i] = 0.0 if lab == 'Ø' else 1.0
            #print(dim, lab, embeds[-1])
        #embed = torch.cat(embeds, 0)
        #spec = torch.tensor(specs, dtype=torch.float)
        #if self.pca is not None:
        #    embed = self.pca.transform(embed.unsqueeze(0).data.numpy())
        #    embed = torch.FloatTensor(embed).squeeze(0)
        #    return embed
        #print(morphosyn, embed, spec)
        return embeds

    # xxx not used
    def reduce_dimension(self, dat):
        embed = self.embed
        # Collect unique tags from data, embed each one,
        # perform dimensionality reduction on the result
        tag_types = Counter([x for x in dat.morphosyn])
        print(tag_types)
        for tag in tag_types:
            print(tag, '->', tag_types[tag])
        tags = list(set([x for x in dat.morphosyn]))
        print(f'Number of unique morphological tags: {len(tags)}')
        embeds = [embed(x).unsqueeze(-1) for x in tags]
        M = torch.cat(embeds, 1).t()
        M = M.data.numpy()
        print(f'Input matrix dimensionality: {M.shape}')
        pca = PCA()
        pca.fit(M)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        ndims = len([x for x in cumvar if x < 0.999])
        print(f'Reducing to dimensionality to: {ndims}')
        self.pca = PCA(n_components=ndims)
        self.pca.fit(M)
        self.dmorphosyn = ndims


class HebrewEmbedder(MorphosynEmbedder):

    def __init__(self):
        dim2vals = {\
            'tense': ['BEINONI', 'FUTURE', 'IMPERATIVE', 'INFINITIVE', 'PAST', 'PRESENT'],\
            'person': ['E', 'FIRST', 'SECOND', 'THIRD'],\
            'gender': ['E', 'F', 'M', 'MF'],\
            'number': ['E', 'PLURAL', 'SINGULAR'],\
            'complete': ['COMPLETE', 'MISSING']             \
        }
        dims = ['tense', 'person', 'gender', 'number', 'complete']
        val2dim = {y: x for x in dim2vals for y in dim2vals[x]}
        dim2embed, dmorphosyn = {}, 0
        for dim in dims:
            n = len(dim2vals[dim])
            dim2embed[dim] = torch.eye(n)
            dmorphosyn += n

        self.dims = dims
        self.dim2vals = dim2vals
        self.val2dim = val2dim
        self.dim2embed = dim2embed
        self.dmorphosyn = dmorphosyn + 1
        print('dimensionality of morphological embedding:', dmorph)

    def morphosyn2tags(self, morphosyn):
        dims, dim2vals, dim2embed = self.dims, self.dim2vals, self.dim2embed
        tags = zip(dims, morphosyn.split('+'))
        tags = [(x, re.sub('.*=', '', y)) for (x, y) in tags]
        return tags

    def embed(self, morphosyn):
        tags = morphosyn2tags(morphosyn)
        embeds = [
            torch.ones(1),
        ]
        for (dim, val) in tags:
            indx = dim2vals[dim].index(val)
            embeds.append(dim2embed[dim][:, indx])
        embed = torch.cat(embeds, 0)
        return embed


class HindiEmbedder(MorphosynEmbedder):

    def __init__(self):
        dim2labels = {\
            'case': ['direct', 'oblique', 'vocative'],\
            'number': ['singular', 'plural'],\
            'gender': ['masculine', 'feminine', 'neuter']             \
        }
        dims = ['case', 'number', 'gender']
        label2dim = {y: x for x in dim2labels for y in dim2labels[x]}
        dim2embed = {}
        dmorphosyn = 0
        for dim in dims:
            n = len(dim2labels[dim])
            dmorphosyn += n
            dim2embed[dim] = torch.cat([\
                torch.eye(n)
                ], 1)
        print('dimensionality of morphological embedding:', dmorphosyn)

        self.dims = dims
        self.dim2labels = dim2labels
        self.label2dim = label2dim
        self.dim2embed = dim2embed
        self.dmorphosyn = dmorphosyn + 1

    def embed(self, morphosyn):
        dims, dim2labels, dim2embed = self.dims, self.dim2labels, self.dim2embed
        tags = zip(dims, morphosyn.split(':'))
        tags = [(x, re.sub('.*=', '', y)) for (x, y) in tags]
        tags = {x[0]: x[1] for x in tags}
        embeds = [
            torch.ones(1),
        ]
        for dim in dims:
            lab = tags[dim] if dim in tags else None
            indx = -1 if lab is None else dim2labels[dim].index(lab)
            embeds.append(dim2embed[dim][:, indx])
        return torch.cat(embeds, 0)


class OneHotEmbedder(MorphosynEmbedder):

    def __init__(self, tags):
        self.tags = tags
        self.ntag = len(tags)
        self.dmorphosyn = self.ntag + 1
        self.embedding = torch.cat(
            [torch.ones(1, self.ntag),
             torch.eye(self.ntag)])
        #print(self.embedding, self.embedding.size())

    def embed(self, morphosyn):
        indx = self.tags.index(morphosyn)
        return self.embedding[:, indx]


class DefaultEmbedder(MorphosynEmbedder):

    def __init__(self):
        self.ndim = 1
        self.dmorphosyn = 1
        self.dim2size = [
            1,
        ]
        self.embed_ = torch.ones(1)
        self.spec_ = torch.ones(1)
        self.dims = ['default']
        self.dim2embed = {'default': torch.ones(1, 1)}

    def embed(self, morphosyn):
        return self.embed_, self.spec_


# unit test
if False:
    datfile = '/Users/colin/Dropbox/TensorProductStringToStringMapping/conll-sigmorphon2018/task1/all/romanian-train-high'
    data = pd.read_table(datfile, header=None)
    data.columns = ['stem', 'output', 'morph']
    MorphosynEmbedder = MorphosynEmbedder.get_embedder(None, data)
