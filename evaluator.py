#!/usr/bin/env python
# -*- coding: utf-8 -*-

def get_accuracy(dat, affixer, decoder, exact_only=True):
    stems, morphs, targs, Stems, Morphs, Targs, targ_len =\
        make_batch(dat, nbatch=len(dat), start_index=0)
    tpr.recorder = Recorder()
    output, _, _ = affixer(Stems, Morphs, max_len=tpr.nrole)
    pretty_print(affixer, Targs)
    output = decoder.decode(output)
    dat['pred'] = [tpr.seq_embedder.idvec2string(x) for x in output]
    tpr.recorder.dump(save=1, test=dat)
    n = len(dat)
    accuracy, avg_mismatch = 0.0, 0.0
    accuracy_change, n_change = 0.0, 0.0
    errors = []
    for i in range(n):
        stem = stems[i]
        targ = targs[i]
        morph = morphs[i]
        out = output[i]
        out = tpr.seq_embedder.idvec2string(out)
        #print(stems[i], outi, '['+ targi +']')
        if out == targ:
            accuracy += 1.0
        else:
            #print('xxx'+ outi +'xxx =/= xxx'+ targi +'xxx')
            errors.append((stem, morph, targ, out))
        if out != stem:
            n_change += 1.0
            if out == targ: accuracy_change += 1.0
        if exact_only:
            continue
        out = out.split(' ')
        targ = targ.split(' ')
        nout, ntarg = len(out), len(targ)
        p = nout if nout<=ntarg else ntarg
        avg_mismatch += np.sum([1.0 if targ[j] != out[j] else 0.0 for j in range(p)])
        avg_mismatch += np.abs(nout - ntarg)
    accuracy = np.round(accuracy / float(n), 3)
    accuracy_change = np.round(accuracy_change / n_change, 3)
    if exact_only:
        return accuracy, accuracy_change, errors
    avg_mismatch = np.round(avg_mismatch / float(n), 3)
    #print(errors)
    tpr.recorder = None
    return accuracy, accuracy_change, avg_mismatch, 
    
        # accuracy on test and train data
        print('evaluating ...')
        train_acc, train_acc_change, train_dist, train_errors =\
            get_accuracy(train, affixer, decoder, False)
        test_acc, test_acc_change, test_dist, test_errors  =\
            get_accuracy(test, affixer, decoder, False)
        train_error_file = 'train_errors'+ output_suffix + '.csv'
        test_error_file = 'test_errors'+ output_suffix +'.csv'
        textwriter = csv.writer(open(train_error_file, 'w', newline='', encoding='utf-8'))
        for error in train_errors:
            try: textwriter.writerow(error)
            except: print('error in writing', error)
        textwriter = csv.writer(open(test_error_file, 'w', newline='', encoding='utf-8'))
        for error in test_errors:
            try: textwriter.writerow(error)
            except: print('error in writing', error)
        
        print('nepochs:', nepoch, 'train_acc:', train_acc, '('+str(train_acc_change) +')', 'test_acc:', test_acc, '('+ str(test_acc_change)+')')
        return affixer, decoder, train_acc, test_acc