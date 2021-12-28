# -*- coding: utf-8 -*-

import pandas as pd
import glob, re, sys
from os import path

# Read unimorph schema
schemafile = path.join(
    path.dirname(__file__), '../features/unimorph_schema.txt')
schema = pd.read_csv(
    schemafile, delimiter='\t', usecols=['Dimension', 'Feature', 'Label'])
schema['Dimension'] = [x.lower() for x in schema['Dimension']]
schema['Feature'] = [x.lower() for x in schema['Feature']]
schema['Label'] = [x.lower() for x in schema['Label']]
#print schema.head()

# Map dimension_i => {(label_ij, feature_ij)}
dimensions = {}
for d in set([x for x in schema['Dimension']]):
    d_features = schema[schema['Dimension'] == d]['Feature']
    d_labels = schema[schema['Dimension'] == d]['Label']
    dimensions[d] = {x: y for (x, y) in zip(d_labels, d_features)}
if 0:  # Print formatted schema
    for key, val in dimensions.items():
        print(key)
        print(val)
        print()

# Map label_ij => dimension_i
label_map = {l: d for d in dimensions for l in dimensions[d]}
#for key,val in label_map.items():
#    print(key)
#    print(val)
#print()

# Convert unimorph tag to dimension=label format
labels_without_dimensions = set([])


def tag2keyvalue(morph):
    global labels_without_dimensions
    #print morph, '->',
    specs = morph.lower().split(';')
    #specs = [re.sub('[/+].*', '', x) for x in specs]
    #specs = [re.sub('(.)[0-9]+$', '\\1', x) for x in specs]
    dims = [label_map[x] if x in label_map else '???' for x in specs]
    labels_without_dimensions |= \
        {specs[i] for i in range(len(specs)) if dims[i]=='???'}
    dimlab = [x + '=' + y for x, y in zip(dims, specs)]
    dimlab = ';'.join(dimlab)
    return dimlab


# Check conversion of all tags in conll2018-sigmorphon data
if 0:
    print(tag2keyvalue('ADJ;NOM/ACC;FEM;SG;DEF'))
    datdir = '/Users/colin/Dropbox/TensorProductStringToStringMapping/conll-sigmorphon2018/task1/all'
    print('file,missing labels')
    for datfile in glob.glob(datdir + '/*'):
        dat = pd.read_csv(datfile, header=None)
        dat.columns = ['stem', 'output', 'morph']
        dat['morph'] = [x.lower() for x in dat['morph']]
        morph2 = [tag2keyvalue(x) for x in dat['morph']]
        if len(labels_without_dimensions) > 0:
            print(
                re.sub('.*/', '', datfile) + ',' +
                ';'.join(labels_without_dimensions))
        labels_without_dimensions = set([])
