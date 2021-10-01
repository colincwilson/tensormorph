#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environ import config
from dataset import DataBatch, write_batch
from recorder import Recorder


def evaluate(model, data):
    print('evaluating ...')
    train_all   = data.get_batch('train_all')
    test_all    = data.get_batch('test_all')

    train_pred  = predict_batch(model, train_all)
    test_pred   = predict_batch(model, test_all)
    train_acc   = evaluate_batch(train_pred)
    test_acc    = evaluate_batch(test_pred)

    # Record processing for initial test item
    config.recorder = Recorder()
    pred, _, _ = model(test_all.Stems.narrow(0, 0, 1),
        test_all.Morphs.narrow(0, 0, 1), test_all.max_len)

    # todo: relocate creation and recording of target-output tpr
    target0 = test_all.outputs[0]
    Target0 = config.form_embedder.string2tpr(target0, False).unsqueeze(0)
    config.recorder.set_values('root', {'target_tpr': Target0})

    config.recorder.dump(save=True)
    config.recorder = None

    write_batch(train_pred, config.save_dir+'/train_pred.csv')
    write_batch(test_pred, config.save_dir+'/test_pred.csv')
    print('train_acc:', train_acc, 'test_acc:', test_acc)
    return (train_pred, test_pred)


def predict_batch(model, batch):
    preds, _, _ =\
        model(batch.Stems, batch.Morphs, batch.max_len)
    preds = config.decoder.decode(preds)
    preds = [config.form_embedder.idvec2string(x) for x in preds]
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
