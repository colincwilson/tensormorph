#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
from data import DataBatch

def evaluate(model, data):
    print ('evaluating ...')
    train_acc = evaluate_batch(model, data.get_batch('train_all'))
    test_acc = evaluate_batch(model, data.get_batch('test_all'))    
    print ('train_acc:', train_acc, 'test_acc:', test_acc)

# xxx calculate levenshtein distance in case of error
def evaluate_batch(model, batch):
    pred, _, _ =\
        model(batch.Stems, batch.Morphs, max_len=20)
    pred = config.decoder.decode(pred)
    pred = [config.seq_embedder.idvec2string(x) for x in pred]
    n, accuracy = len(batch.targs), 0.0
    for i,targ in enumerate(batch.targs):
        #print (pred[i], ' =?= ', targ)
        if pred[i] == targ: accuracy += 1.0
    return (accuracy / n)