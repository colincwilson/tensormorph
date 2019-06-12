#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
from .tpr import *
from .dataset import DataBatch
#from model import Model
import csv  # todo: replace with pandas

class Trainer():
    def __init__(self, model):
        self.model  = model

        optimizer = optim.Adagrad # optim.RMSprop # optim.Adadelta
        self.model_optim =\
            optimizer(model.parameters(), config.learn_rate, config.dc)
        self.decoder_optim =\
            optimizer(config.decoder.parameters(), config.learn_rate, config.dc)
        self.criterion =\
            nn.CrossEntropyLoss(ignore_index=0, reduction='none')\
            if config.loss_func=='loglik' else nn.MSELoss(reduction='none')
        #self.regularizer = nn.MSELoss(size_average=False)
        self.regularizer = nn.L1Loss(size_average=False)

        print ('number of trainable parameters: ',\
            count_parameters(model), '+',
            count_parameters(config.decoder)
        )


    def train(self, data):
        print ('training ...')
        for epoch in range(config.nepoch):
            loss = self.train1(data, epoch)
            if loss.item() < 0.01:
                break
        return None


    def train1(self, data, epoch, gradient_update=True):
        model       = self.model
        decoder     = config.decoder
        regularizer = self.regularizer

        # sample minibatch and predict outputs
        batch = data.get_batch(nbatch=config.batch_size)
        Stems, Morphs, Outputs = batch.Stems, batch.Morphs, batch.Outputs
        max_len = 20
        pred, affix, (pivot, copy_stem, unpivot, copy_affix) =\
            model(Stems, Morphs, max_len=max_len)

        # reset gradients -- this is very important
        model.zero_grad()
        decoder.zero_grad()

        # get total log-likelihood loss and value for each batch member
        loss, losses =  self.loglik_loss(pred, Outputs, max_len)

        # regularize affix toward epsilon, disprefer pivoting,
        # prefer stem copying
        lambda_morph = 1.0e-4
        lambda_phon = 1.0e-0
        loss += lambda_morph * regularizer(affix, torch.zeros_like(affix))
        loss += lambda_morph * regularizer(pivot, torch.zeros_like(pivot))
        loss += lambda_morph * regularizer(copy_stem, torch.ones_like(copy_stem))
        for W in model.phono_rules.parameters():
            loss += lambda_phon * regularizer(W, torch.zeros_like(W))
        #loss  += lambda_reg * regularizer(output, torch.zeros_like(output))

        # report current state
        if (epoch % 50 == 0):
            print (epoch, 'loss =', loss.item())
            self.report(model, decoder, Stems, Morphs, Outputs)

        # update parameters
        if gradient_update:
            loss.backward()
            self.model_optim.step()
            self.decoder_optim.step()
        
        return loss


    def report(self, affixer, decoder, Stems, Morphs, Outputs):
        config.recorder = Recorder()
        pred, _, _ =\
            affixer(Stems.narrow(0,0,1), Morphs.narrow(0,0,1), max_len=20) 
        pretty_print(affixer, Outputs)
        print('tau_morph =',
            np.round(affixer.combiner.morph_attender.tau.data[0], 4))
        print('tau_posn =', 
            np.round(affixer.combiner.posn_attender.tau.data[0], 4))
        print('tau_decode =',
            np.round(decoder.tau.data[0], 4))
        #print('pivoter W0 =', affixer.pivoter.W0.weight.data.numpy())
        #print('pivoter bias0 =', affixer.pivoter.W0.bias.data.numpy())
        #print('pivoter W1 =', affixer.pivoter.W1.weight.data.numpy())
        #print('pivoter bias1 =', affixer.pivoter.W1.bias.data.numpy())
        #print('pivoter a =', affixer.pivoter.a.data.numpy())
        config.recorder = None
        return None


    def loglik_loss(self, pred, Outputs, max_len):
        # get loss vector for each batch member
        sim    = config.decoder(pred)   # xxx rename lhs
        losses = self.criterion(sim, Outputs)

        # sum losses over positions within each output,
        # average over members of minibatch
        loss = torch.sum(losses, 1)
        loss = torch.mean(loss)

        return loss, losses


# count number of trainable parameters
def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# report current processing for one example
# xxx move
def pretty_print(affixer, outputs, header='**root**'):
    print('\n'+header)
    record = config.recorder.dump()
    stem = config.decoder.decode(record['root-stem_tpr'])[0]
    affix = config.decoder.decode(record['root-affix_tpr'])[0]
    pred = config.decoder.decode(record['root-output_tpr'])[0]
    output_segs = 'NA' if outputs is None else\
        config.seq_embedder.idvec2string(outputs[0,:].data.numpy())
    copy_stem = record['root-copy_stem'].data[0,:]
    copy_affix = record['root-copy_affix'].data[0,:]
    pivot = record['root-pivot'].data[0,:]
    unpivot = record['root-unpivot'].data[0,:]
    #morph_indx = record['root-morph_indx'].data[0,:,0]

    stem_segs = config.seq_embedder.idvec2string(stem)
    stem_annotated = config.seq_embedder.idvec2string(stem, copy_stem, pivot)
    affix_annotated = config.seq_embedder.idvec2string(affix, copy_affix, unpivot)
    pred_segs = config.seq_embedder.idvec2string(pred)

    print(stem_segs, '    ', output_segs, '    ', pred_segs)
    print('annotated stem:', stem_annotated)
    print('annotated affix:', affix_annotated)
    print('copy_stem:', np.round(copy_stem.data.numpy(), 2))
    print('pivot:', np.round(pivot.data.numpy(), 2))
    print('copy_affix:', np.round(copy_affix.data.numpy(), 2))
    print('unpivot:', np.round(unpivot.data.numpy(), 2))

    if 0: # print affix tpr, etc.
        print(np.round(record['root-affix_tpr'].data[0,:,1].numpy(), 2))
        print(np.round(record['root-output_tpr'].data[0,:,0].numpy(), 2))
        print(np.round(record['root-output_tpr'].data[0,:,1].numpy(), 2))
        pred_prob = config.decoder(record['root-output_tpr'])
        pred_prob = torch.exp(log_softmax(pred_prob, 1))
        print(np.round(pred_prob.data[0,:,1].numpy(), 3))
    
    #print(np.round(record['root-affix_tpr'].data[0,0,:].numpy(), 2))
    #print('morph_indx:', np.round(morph_indx.data.numpy(), 2))
    #if affixer.redup:
    #    pretty_print(affixer.affixer, None, header='**reduplicant**')
    #    print('\n')