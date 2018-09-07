#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr
from tpr import *
from seq_embedder import *
from decoder import Decoder, LocalistDecoder
from affixer import Affixer
#from redup import Reduplicator
from ensemble import EnsembleAffixer
from entropy import binary_entropy
from sklearn.model_selection import train_test_split
import csv

# count number of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# train and test split
def make_split(dat, test_size=0.25):
    train, test = train_test_split(dat, test_size=test_size)
    return train, test


# batch of training examples
def make_batch(dat, nbatch=20, debug=0, start_index=None):
    seq_embedder   = tpr.seq_embedder
    morph_embedder = tpr.morph_embedder

    if start_index is None: batch = dat.sample(nbatch)
    else: batch = dat[start_index:(start_index+nbatch-1)]

    stems  = [x for x in batch['stem']]
    stems  = [string2delim(x) for x in stems]
    morphs = [x for x in batch['morph']]
    targs  = [x for x in batch['output']]
    targs  = [string2delim(x) for x in targs]

    Stems = torch.zeros((nbatch, tpr.dfill, tpr.nrole))
    for i,stemi in enumerate(stems):
        try: Stems[i,:,:] = seq_embedder.string2tpr(stemi, False)
        except:
            print 'error in embedding stem', stemi
            sys.exit(0)
        if debug:
            print '__'+stemi+'__', '->',

    Morphs = torch.zeros((nbatch, tpr.dmorph))
    for i, morphi in enumerate(morphs):
        #Morphs.data[i,:] = morph_embedder.embed(morphi)
        Morphs[i,:] = morph_embedder.embed(morphi)
        #print morphi, '->', Morphs.data[i,:]

    targ_len = [0]*nbatch
    Targs = torch.LongTensor(nbatch, tpr.nrole)
    Targs.fill_(0)  # padd with ε (by convention the 0th symbol)
    for i,targi in enumerate(targs):
        x = seq_embedder.string2idvec(targi, False)
        targ_len[i] = len(x)
        #Targs.data[i,0:len(x)] = torch.LongTensor(x)
        Targs[i,0:len(x)] = torch.LongTensor(x)
        if debug:
            print '__'+targi+'__'

    max_targ_len = np.max(np.array([len(x.split(' ')) for x in targs]))
    #print 'max target length:', max_targ_len
    if max_targ_len >= tpr.nrole:
        print 'error: max target length >= nrole'
        sys.exit(0)
    return stems, morphs, targs, Stems, Morphs, Targs, targ_len


def get_accuracy(dat, affixer, decoder, exact_only=True):
    stems, morphs, targs, Stems, Morphs, Targs, targ_len = make_batch(dat, len(dat))
    tpr.record = True
    output, _, _ = affixer(Stems, Morphs, max_len=tpr.nrole)
    # save recordings for off-line visualization
    stem_tpr    = affixer.recorder.record['stem_tpr'].select(0,0).data.numpy()
    affix_tpr   = affixer.recorder.record['affix_tpr'].select(0,0).data.numpy()
    stem_tpr    = np.clip(stem_tpr, 1.0e-8, 1.0e8)
    affix_tpr   = np.clip(affix_tpr, 1.0e-8, 1.0e8)
    np.save('/Users/colin/Desktop/writer_outputs/stem.npy', stem_tpr)
    np.save('/Users/colin/Desktop/writer_outputs/affix.npy', affix_tpr)
    writer_dump = affixer.combiner.writer.recorder.dump()
    for x,y in writer_dump.items():
        y = y.select(0,0).data.numpy()
        y = np.clip(y, 1.0e-8, 1.0e8)
        np.save('/Users/colin/Desktop/writer_outputs/' + x, y)

    pretty_print(affixer, Targs)
    output = decoder.decode(output)
    accuracy, avg_mismatch, n = 0.0, 0.0, len(dat)
    accuracy_change, n_change = 0.0, 0.0
    errors = []
    for i in xrange(n):
        stem = stems[i]
        targ = targs[i]
        morph = morphs[i]
        out = output[i]
        out = tpr.seq_embedder.idvec2string(out)
        #print stems[i], outi, '['+ targi +']'
        if out == targ:
            accuracy += 1.0
        else:
            #print 'xxx'+ outi +'xxx =/= xxx'+ targi +'xxx'
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
        avg_mismatch += np.sum([1.0 if targ[j] != out[j] else 0.0 for j in xrange(p)])
        avg_mismatch += np.abs(nout - ntarg)
    accuracy = np.round(accuracy / float(n), 3)
    accuracy_change = np.round(accuracy_change / n_change, 3)
    if exact_only:
        return accuracy, accuracy_change, errors
    avg_mismatch = np.round(avg_mismatch / float(n), 3)
    #print errors
    tpr.record = False
    return accuracy, accuracy_change, avg_mismatch, errors


def pretty_print(affixer, targs, header='**root**'):
    print '\n'+header
    record = dict(affixer.recorder.dump().items() +\
         affixer.combiner.recorder.dump().items())
    stem_tpr = tpr.decoder.decode(record['stem_tpr'])[0]
    affix_tpr = tpr.decoder.decode(record['affix_tpr'])[0]
    output = tpr.decoder.decode(record['output_tpr'])[0]
    targ = 'NA' if targs is None else tpr.seq_embedder.idvec2string(targs[0,:].data.numpy())
    copy = record['copy'][0,:]
    pivot = record['pivot'][0,:]
    unpivot = record['unpivot'][0,:]
    morph_indx = record['morph_indx'].data[0,:,0]

    stem = tpr.seq_embedder.idvec2string(stem_tpr)
    stem_annotated = tpr.seq_embedder.idvec2string(stem_tpr, copy, pivot)
    affix_annotated = tpr.seq_embedder.idvec2string(affix_tpr, None, unpivot)
    output = tpr.seq_embedder.idvec2string(output)

    print stem, u'    ', targ, u'    ', output
    print 'annotated stem:', stem_annotated
    print 'annotated affix:', affix_annotated
    print 'copy:', np.round(copy.data.numpy(), 2)
    print 'pivot:', np.round(pivot.data.numpy(), 2)
    print 'unpivot:', np.round(unpivot.data.numpy(), 2)
    print 'morph_indx:', np.round(morph_indx.data.numpy(), 2)

    #if affixer.redup:
    #    pretty_print(affixer.affixer, None, header='**reduplicant**')
    #    print '\n'


class Trainer():
    def __init__(self, redup=False, lr=0.1, dc=0.0, verbosity=1):
        print 'Trainer.init()'
        self.redup, self.lr, self.dc, self.verbosity =\
        redup, lr, dc, verbosity
        self.affixer = Affixer(redup); self.affixer.init()
        #self.affixer = EnsembleAffixer(redup, True, 2); self.affixer.init()
        self.decoder = Decoder() if 0 else LocalistDecoder()
        tpr.decoder = self.decoder

        print 'number of trainable parameters =', count_parameters(self.affixer), '+', count_parameters(self.decoder)

        optimizer = optim.Adagrad # optim.RMSprop
        self.affixer_optim = optimizer(self.affixer.parameters(), lr, dc)
        self.decoder_optim = optimizer(self.decoder.parameters(), lr, dc)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')\
                        if tpr.loss_func=='loglik' else nn.MSELoss(reduction='none')
        self.regularizer = nn.MSELoss()

    def train_and_test(self, train, test, nbatch, max_epochs):
        affixer, decoder = self.affixer, self.decoder

        print 'training ...',
        nepoch = 0
        for epoch in xrange(max_epochs):
            loss, train_acc = self.train_and_test1(train, test, nbatch=nbatch, epoch=epoch, gradient_update=True)
            if loss.item() / float(nbatch) < 0.01 and train_acc==1.0:
                nepoch = epoch
                break

        # accuracy on test and train data
        print 'evaluating ...'
        train_acc, train_acc_change, train_dist, train_errors = get_accuracy(train, affixer, decoder, False)
        test_acc, test_acc_change, train_dist, test_errors  = get_accuracy(test,  affixer, decoder, False)
        textwriter = csv.writer(open('train_errors.csv', 'wb'))
        for error in train_errors:
            try:
                row = [s.encode('utf-8') for s in error]
                textwriter.writerow(row)
            except:
                print 'error in writing', error
        textwriter = csv.writer(open('test_errors.csv', 'wb'))
        for error in test_errors:
            try:
                row = [s.encode('utf-8') for s in error]
                textwriter.writerow(row)
            except:
                print 'error in writing', error
        
        print 'nepochs:', nepoch, 'train_acc:', train_acc, '('+str(train_acc_change) +')', 'test_acc:', test_acc, '('+ str(test_acc_change)+')'
        return affixer, decoder


    def train_and_test1(self, train, test, nbatch, epoch=0, gradient_update=True):
        affixer, decoder, affixer_optim, decoder_optim, criterion, regularizer =\
        self.affixer, self.decoder, self.affixer_optim, self.decoder_optim, self.criterion, self.regularizer

        # sample min-batch and predict outputs
        stems, morphs, targs, Stems, Morphs, Targs, targ_len = make_batch(train, nbatch)
        max_len = 20 # max(targ_len)
        output, affix, (pivot, copy, unpivot) =\
            self.affixer(Stems, Morphs, max_len=max_len)
        #print Targs.data.numpy()

        # xxx testing
        if 0:
            affixer.pivoter.a.data[:] = 10.0
            pivot = affixer.pivoter(Stems)
            print pivot.data[0,:].numpy()
            sys.exit(0)

        # reset gradients -- this is very important
        self.affixer.zero_grad()
        self.decoder.zero_grad()

        # get total log-likelihood loss and value for each batch member
        loss, losses =  self.loglik_loss(output, Targs, max_len) if tpr.loss_func=='loglik'\
                        else self.euclid_loss(output, targs, max_len)

        # regularize affix toward epsilon
        loss += 1.0e-3 * regularizer(affix, torch.zeros_like(affix))

        # regularize pivot, copy, unpivot toward extremes (min-entropy)
        loss += 1.0e-3 * binary_entropy(pivot).sum()
        loss += 1.0e-3 * binary_entropy(unpivot).sum()
        loss += 1.0e-3 * binary_entropy(copy).sum()

        train_acc = 0.0
        if (epoch % 50 == 0) and self.verbosity>0:
            print
            tpr.record = True
            avg_loss = loss.item() / float(nbatch)
            #train_acc, errors = get_accuracy(train, affixer, decoder)
            train_acc, errors = None, None
            output, _, _ = affixer(Stems[0,:,:].unsqueeze(0), Morphs[0,:].unsqueeze(0), tpr.nrole)
            print epoch,  'avg loss =', avg_loss, 'train_acc =', train_acc
            pretty_print(affixer, Targs)
            if True:
                print 'tau_morph =', np.round(affixer.combiner.morph_attender.tau.data[0], 4)
                print 'tau_posn =',  np.round(affixer.combiner.posn_attender.tau.data[0], 4)
                print 'tau_decode =', np.round(decoder.tau.data[0], 4)
                #print 'pivoter W0 =', affixer.pivoter.W0.weight.data.numpy()
                #print 'pivoter bias0 =', affixer.pivoter.W0.bias.data.numpy()
                #print 'pivoter W1 =', affixer.pivoter.W1.weight.data.numpy()
                #print 'pivoter bias1 =', affixer.pivoter.W1.bias.data.numpy()
                #print 'pivoter a =', affixer.pivoter.a.data.numpy()
            tpr.record = False

        # calculate gradient and update parameters
        if gradient_update:
            loss.backward()
            affixer_optim.step()
            decoder_optim.step()

        # manually raise minimum rbf and decoding precision
        # after half of the epochs have been completed
        #tpr.tau_min.data[:] += 0.01

        return loss / float(nbatch), train_acc


    def loglik_loss(self, output, Targs, max_len):
        # get loss vector for each batch member
        sim    = self.decoder(output)
        losses = self.criterion(sim, Targs)

        # sum losses over positions within output,
        # average over batch members
        loss = torch.sum(losses, 1)
        loss = torch.mean(loss)

        # debugging
        losses_bad = np.isnan(losses.data.numpy())
        if np.any(losses_bad):
            m, n = losses_bad.shape
            for i in xrange(m):
                losses_bad_i = losses_bad[i,:]
                if not np.any(losses_bad_i):
                    continue
                for k in xrange(n):
                    if losses_bad_i[k]:
                        print 'bad loss for symbol @'+ str(i) + str(k) + str(Targs[i,k])
            sys.exit(0)

        return loss, losses


    def euclid_loss(self, output, targs, max_len):
        nbatch = len(targs)
        Targs_ = torch.zeros(nbatch, tpr.dfill, tpr.drole)
        for i,targi in enumerate(targs):
            X = tpr.seq_embedder.string2tpr(targi, False)
            Targs_[i,:,:] += X

        losses = self.criterion(output, Targs_)

        # sum losses over positions within output,
        # average over batch members
        loss = torch.sum(losses, 1)
        loss = torch.mean(loss)

        #print losses
        #print losses.shape
        #print loss
        return loss, losses