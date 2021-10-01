#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from data_util import morph_batcher
from environ import config
from tpr import *
from combiner import Combiner
from writer import Writer

class Aligner(pl.LightningModule):
    """
    Search for alignments with generic LSTM
    """
    def __init__(self, batch=None, reduplication=False):
        super(Aligner, self).__init__()
        self.batch = batch # fixed batch for this aligner
        self.reduplication = reduplication
        self.combiner = Combiner()
        self.writer = Writer()
        self.decoder = config.decoder
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0, reduction='none')
        self.combiner.morph_attender.tau.clamp(3.0,3.0)
        self.combiner.posn_attender.tau.clamp(3.0,3.0)
        #self.decoder.tau.clamp(0.0,10.0)

        nbatch = batch['Stem'].shape[0]
        self.nhidden = 10
        self.lstm = torch.nn.GRUCell(
            input_size = 1,
            hidden_size = self.nhidden)
       #     hidden_size = self.nhidden,
       #     num_layers = 1,
       #     batch_first = True,
       #     dropout = 0,
       #     bidirectional = False)
        self.alpha = torch.nn.Linear(
            in_features = self.nhidden,
            out_features = 1)
        self.beta0 = torch.nn.Linear(
            in_features = self.nhidden,
            out_features = 1) #config.nrole)
        self.beta1 = torch.nn.Linear(
            in_features = self.nhidden, #self.nhidden,
            out_features = 1) #config.nrole)
        self.omega = torch.nn.Linear(
            in_features = self.nhidden,
            out_features = 1) #config.nrole)
        if not self.reduplication:
            self.Affix = Parameter(
                torch.randn((config.dfill, config.drole)),
                requires_grad = True)
            self.Affix.data[0,:].fill_(1.0) # all wildcards □
            self.Affix[0,:].clamp(1.0,1.0)


    def forward(self, batch):
        Stem, Morphosyn, max_len = \
            batch['Stem'], batch['Morphosyn'], batch['max_len']
        nbatch = Stem.shape[0]
        if self.reduplication:
            Affix = batch['Stem']
        else:
            Affix = self.Affix \
                .view(config.dfill, config.nrole) \
                .unsqueeze(0) \
                .expand(nbatch, -1, -1)
            #Affix = torch.tanh(Affix)
        
        self.writer.init(nbatch)
        copy_stem = torch.ones((nbatch, 1), requires_grad=False)
        copy_affix = torch.ones((nbatch, 1), requires_grad=False)
        ones = torch.ones( # dummy LSTM input
            (nbatch, 1), 
            requires_grad=False)
        h = None
        for i in range(config.nrole + 2):
            # Update hidden
            if h is None:
                h = self.lstm(ones)
            else:
                h = self.lstm(ones, h)
            # Project hidden to scalar indices
            a = self.alpha(h)
            b0 = self.beta0(h)
            b1 = self.beta1(h)
            c = self.omega(h)
            # Map scalar indices to attn distributions
            alpha = self.combiner.morph_attender(a)
            beta0 = self.combiner.posn_attender(b0)
            beta1 = self.combiner.posn_attender(b1)
            omega = self.combiner.posn_attender(c)            
            # Project hidden to softmax attention
            #alpha = softmax(self.alpha(h), -1)
            #beta0 = softmax(self.beta0(h), -1)
            #beta1 = softmax(self.beta1(h), -1)
            #omega = softmax(self.omega(h), -1)
            #print(alpha.shape, beta0.shape,
            #    beta1.shape, omega.shape)
            # Update output tpr
            self.writer(Stem, Affix, alpha, beta0, beta1, omega, 
                copy_stem, copy_affix)
        Pred = self.writer.normalize()
        return Pred


    def training_step(self, batch, batch_nb):
        batch = self.batch # xxx ignore input
        batch['Pred'] = self.forward(batch)
        loss, losses = self.loglik_loss(
            batch['Pred'], batch['Output'])
        if batch_nb == 0:
            #print(f'copy_stem: {torch.sigmoid(self.copy_stem[0])}')
            #print(f'copy_affix: {torch.sigmoid(self.copy_affix[0])}')
            #print(f'pivot: {torch.sigmoid(self.pivot[0])}')
            #print(f'unpivot: {torch.sigmoid(self.unpivot[0])}')
            #print(f'tau_morph: {self.combiner.morph_attender.tau}')
            #print(f'tau_posn: {self.combiner.posn_attender.tau}')
            print(f'tau_decode: {config.decoder.tau}')
            print(f'loss: {loss}')
            if not self.reduplication:
                Affix = self.Affix \
                    .view(config.dfill, config.nrole)
                Affix = torch.tanh(Affix)
                affix = config.form_embedder.tpr2string(Affix)
                print(f'affix: {affix}')
            pred = config.form_embedder.tpr2string(batch['Pred'][0])
            print(f'pred: {pred}')
        print(loss)
        return {'loss': loss}
    

    def configure_optimizers(self):
        # ex. Adagrad lr = 1.0e-1, lr_decay = 0.0, weight_decay small
        optimizer = optim.Adagrad(
            self.parameters(),
            lr = config.learn_rate,
            lr_decay = 1.0e-3,
            weight_decay = config.dc)
        return optimizer


    def train_dataloader(self):
        return DataLoader(
            config.train_dat,
            batch_size = config.batch_size,
            shuffle = True,
            collate_fn = morph_batcher
        )


    def loglik_loss(self, Pred, Output):
        # Get loss vector for each batch member
        #print(Pred.shape, Output.shape)
        logprobs = self.decoder(Pred)
        losses = self.criterion(logprobs, Output)
        loss = torch.sum(losses, 1)
        loss = torch.sum(loss)
        return loss, losses


    def train(self):
        """
        Train alignments
        """
        trainer = Trainer(
            logger = False,
            show_progress_bar = False,
    #        min_nb_epochs = config.min_epoch, # deprecated
            min_epochs = config.min_epochs,
            max_epochs = config.max_epochs,
            gradient_clip_val = 5.0, # xxx make config option
            #gpus = 1,
            auto_lr_find = False
            )
        trainer.fit(self)

