#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
from data import DataBatch

def evaluate(model, data):
    print ('evaluating ...')
    train_pred  = predict_batch(model, data.get_batch('train_all'))
    test_pred   = predict_batch(model, data.get_batch('test_all'))

    train_acc   = evaluate_batch(train_pred)
    test_acc    = evaluate_batch(test_pred)
    print ('train_acc:', train_acc, 'test_acc:', test_acc)
    return (train_pred, test_pred)


def predict_batch(model, batch):
    preds, _, _ =\
        model(batch.Stems, batch.Morphs, max_len=20)
    preds = config.decoder.decode(preds)
    preds = [config.seq_embedder.idvec2string(x) for x in preds]
    batch = batch._replace(preds = preds)
    return batch


# todo: calculate levenshtein distance in case of error
def evaluate_batch(batch):
    outputs  = batch.outputs
    preds    = batch.preds
    accuracy = 0.0
    n        = len(outputs)
    for i,output in enumerate(outputs):
        if preds[i]==output:
            accuracy += 1.0
    return (accuracy / n)
