#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, re, sys
from tensormorphy import environ, evaluator, phon_features
from tensormorphy.segment_embedder import SegmentEmbedder
from tensormorphy.form_embedder import FormEmbedder
from tensormorphy.dataset import DataSet
from tensormorphy.affixer import Affixer
from tensormorphy.trainer import Trainer
from affix_test_cases import import_data
import pandas as pd
import numpy as np

# parse commandline arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--nbatch',\
    help='Number of <input,output> pairs in each batch')
argparser.add_argument('--nepoch',\
    help='Number of training epochs')
args, residue = argparser.parse_known_args()


# select dataset (xxx make commandline argument)
data_select = ['english_ing', 'english_ness', 'english_un',\
               'english_shm', 'chamorro_um', 'hungarian_dat',\
               'hebrew_paal', 'hindi_nouns', 'maltese', 'conll'][4]
data = import_data(data_select)
data_set = DataSet( data['dat'],
                    data['held_in_stems'],
                    data['held_out_stems'],
                    data['vowels']
                    )

feature_file = '~/Dropbox/TensorProductStringToStringMapping/00features/' +\
    ['hayes_features.csv', 'panphon_ipa_bases.csv'][0]
feature_matrix = phon_features.import_features(feature_file, data_set.segments)

symbol_params = {'feature_matrix': feature_matrix, }
role_params = {'nrole': data_set.max_len+4, }
form_embedder = FormEmbedder(symbol_params, role_params)
environ.init(form_embedder) # makes dummy morphosyn_embedder

data_set.split_and_embed(test_size=0.25)
model = Affixer()
trainer = Trainer(model)
environ.config.nepoch = 1500
trainer.train(data_set)

train_pred, test_pred =\
    evaluator.evaluate(model, data_set)
sys.exit(0)


# # # # # OLD CODE # # # # #


seq_embedder, morph_embedder, train, test = import_data(data_select)
tpr.init(seq_embedder, morph_embedder)

print('filler dimensionality:', tpr.dfill)
print('role dimensionality:', tpr.drole)
print('distributed roles?', tpr.random_roles)

print('train/test split:')
print('\t', len(train), 'training examples')
print('\t', len(test), 'testing examples')


# run trainer
tpr.save_dir = '/Users/colin/Desktop/tmorph_output'
nbatch = min(40,len(train)) if args.nbatch is None else int(args.nbatch)
nepoch = 1000 if args.nepoch is None else int(args.nepoch)
trainer = trainer.Trainer( redup=False, lr=1.0e-1, dc=0.0, verbosity=1 )
affixer, decoder = trainer.train_and_test( train, test, nbatch=nbatch, max_epochs=nepoch )


if False:
    tpr.trace = True
    train = train.iloc[0:2].reset_index()
    test = test.iloc[0:2].reset_index()
    train.stem, train.output = u't r i s t i', u't r u m i s t i'
    trainer.train_and_test1(train, test, nbatch=len(train))
    print(tpr.traces)
    for x in tpr.traces:
        f = '/Users/colin/Desktop/dump/'+ x +'.txt'
        y = tpr.traces[x]
        print(y.__class__.__name__)
        if type(y) is np.ndarray:
            np.savetxt(f, y, delimiter=',')
        else:
            print(x, y)


if False: # test by hand
    trainer.affixer.morph_attender.tau.data[:] = 5.0
    trainer.affixer.posn_attender.tau.data[:] = 5.0
    Stems = string2tpr(u'q a f a ts').unsqueeze(0)
    Affix = string2tpr(u't i o ⋉', False).unsqueeze(0)
    copy = torch.ones(tpr.nrole).unsqueeze(0)
    pivot = torch.zeros(tpr.nrole).unsqueeze(0)
    unpivot = torch.zeros(tpr.nrole).unsqueeze(0)
    copy[0,2] = copy[0,4] = 0.0
    pivot[0,0] = pivot[0,3] = 1.0
    unpivot[0,1] = unpivot[0,2] = 1.0
    test = {\
    'affix': Affix,\
    'copy': copy,\
    'pivot': pivot,\
    'unpivot': unpivot\
    }
    output, traces = trainer.affixer(Stems, 10, True, test)
    stem = trainer.decoder.decode(Stems)[0]
    affix = trainer.decoder.decode(Affix)[0]
    stem = [x+' _' if pivot[0,i]==1.0 else x for i,x in enumerate(stem.split(' '))]
    stem = [x+'/' if copy[0,i]==0.0 else x for i,x in enumerate(stem)]
    affix = [x+' _' if i<25 and unpivot[0,i]==1.0 else x for i,x in enumerate(affix.split(' '))]
    stem = ' '.join(stem)
    affix = ' '.join(affix)
    output = ' '.join(trainer.decoder.decode(output))
    print('stem:', stem)
    print('affix:', affix)
    print(' -> ')
    print('output: ', output)
    for trace in traces:
        print(trace, np.round(traces[trace], 2))
    sys.exit(0)
