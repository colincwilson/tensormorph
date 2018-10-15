#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import glob, re, sys
from os import path

# read unimorph schema
schemafile = path.join(path.dirname(__file__), 'unimorph_schema.txt')
schema = pd.read_table(schemafile, delimiter='\t')
schema['Dimension'] = [x.lower() for x in schema['Dimension']]
schema['Feature'] = [x.lower() for x in schema['Feature']]
schema['Label'] = [x.lower() for x in schema['Label']]
#print schema.head()

# map dimension_i => {(label_ij, feature_ij)}
dimensions = {}
for d in set([x for x in schema['Dimension']]):
    d_features = schema[schema['Dimension']==d]['Feature']
    d_labels = schema[schema['Dimension']==d]['Label']
    dimensions[d] = {x:y for (x,y) in zip(d_labels, d_features)}
#print dimensions

# map label_ij => dimension_i
label_map = {l:d for d in dimensions for l in dimensions[d]}
#print label_map

# convert unimorph tag to dimension=label format
labels_without_dimensions = set([])
def tag2keyvalue(morph):
    global labels_without_dimensions
    #print morph, '->',
    specs = morph.lower().split(';')
    specs0 = [re.sub('[/+].*', '', x) for x in specs]
    specs0 = [re.sub('(.)[0-9]+$', '\\1', x) for x in specs0]
    dims = [label_map[x] if x in label_map else '???' for x in specs0]
    labels_without_dimensions |= \
        {specs[i] for i in range(len(specs)) if dims[i]=='???'}
    dimlab = [x +'='+ y for x,y in zip(dims, specs)]
    dimlab = ';'.join(dimlab)
    return dimlab

# check conversion of all tags in conll2018-sigmorphon data
if False:
    print(tag2keyvalue('ADJ;NOM/ACC;FEM;SG;DEF'))
    datdir = '/Users/colin/Dropbox/TensorProductStringToStringMapping/conll-sigmorphon2018/task1/all'
    print('file,missing labels')
    for datfile in glob.glob(datdir+'/*'):
        dat = pd.read_table(datfile, header=None)
        dat.columns = ['stem', 'output', 'morph']
        dat['morph'] = [x.lower() for x in dat['morph']]
        morph2 = [tag2keyvalue(x) for x in dat['morph']]
        if len(labels_without_dimensions) > 0:
            print(re.sub('.*/', '', datfile) +','+ ';'.join(labels_without_dimensions))
        labels_without_dimensions = set([])
