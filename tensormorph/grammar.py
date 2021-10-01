# -*- coding: utf-8 -*-

import config
from tpr import *
from morph import *
from radial_basis import GaussianPool
from decoder import Decoder
import cogrammar, redup_cogrammar, mixture_cogrammar
from data_util import morph_batcher
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
#from torch.nn import TransformerEncoderLayer
from matcher import LiteralMatcher
from phonology import Phonology
from recorder import assign_node_names, report
import pytorch_lightning as pl
from distance import sqeuclid_batch1
#import editdistance


class Grammar(pl.LightningModule):
    """
    Wrap grammar in pytorch-lightning
    """

    def __init__(self, learning_rate, reduplication=False):
        super(Grammar, self).__init__()
        self.morph_attender = GaussianPool(
            n=2, tau_min=0.5)  # xxx config option
        self.posn_attender = GaussianPool(
            n=config.nrole, tau_min=0.5)  # xxx config option
        self.decoder = Decoder(tau_min=0.5)  # xxx config option
        config.morph_attender = self.morph_attender
        config.posn_attender = self.posn_attender
        config.decoder = self.decoder
        #config.decoder.tau[:] = 1.0 # xxx freeze
        #config.decoder.tau.detach_()

        if reduplication:
            self.cogrammar = redup_cogrammar.RedupCogrammar()
        else:
            self.cogrammar = cogrammar.MultiCogrammar()
            #self.cogrammar = mixture_cogrammar.MixtureCogrammar()

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-1, reduction='none')  # use ignore_index?
        self.learning_rate = learning_rate
        assign_node_names(self, None, 'root')  # xxx necessary?

    def forward(self, batch):
        stem, morphosyn, max_len = \
            batch['stem'], batch['morphosyn'], batch['max_len']
        pred = self.cogrammar(stem, morphosyn, max_len)
        return pred

    def training_step(self, batch, batch_nb):
        batch['pred'] = self.forward(batch)
        loss, losses = self.loss_func(batch['pred'], batch['output'])
        # xxx alternative euclidean loss, endogenous to tprs,
        # xxx effective only with distributed roles
        # nbatch = batch['pred'].form.shape[0]
        #dist = sqeuclid_batch1(batch['pred'].form.view(nbatch, -1),
        #                         batch['output_tpr'].form.view(nbatch, -1))
        # loss = torch.mean(dist)

        if batch_nb == 0:
            report(self, batch)

        # [hack] Stop early when loss gets low, perhaps better way at
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1406
        if loss < 0.001:
            raise KeyboardInterrupt

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        batch['pred'] = self.forward(batch)
        loss, losses = self.loss_func(batch['pred'], batch['output'])
        print(f'dev loss: {np.round(loss.item(), 2)}')
        # todo: also calculate whole-string, levenshtein accuracy
        self.lr_sched.step(loss.item())  # xxx ReduceLROnPlateau only
        return {'loss': loss}

    #def test_step(self, batch, batch_nb):
    # xxx todo

    def configure_optimizers(self):
        # Possibly different optimizer params for phonology module
        # (https://github.com/PyTorchLightning/pytorch-lightning/issues/2005)
        def penalize(n):
            return re.search('phonology.matcher', n)

        params = list(self.named_parameters())  #self.state_dict()
        base_params = [p for n, p in params if not penalize(n)]
        phon_params = [p for n, p in params if penalize(n)]
        optimizer = optim.Adagrad(  # Adagrad, ASGD, Adadelta, Adam, ...
            #self.parameters(),
            [{'params': base_params, 'weight_decay': config.weight_decay}, \
             {'params': phon_params, 'weight_decay': config.weight_decay}],

            lr=self.learning_rate)

        #lr_sched = {'scheduler': lr_scheduler.StepLR(
        #    optimizer, step_size = 1, gamma = 0.75, verbose = 1) }
        #lr_sched = {'scheduler': lr_scheduler.OneCycleLR(
        #    optimizer, max_lr = 1.0, total_steps = config.max_epochs,
        #    cycle_momentum = False, final_div_factor = 100, verbose = 1) }
        self.lr_sched = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.50,  # xxx 1/2 or 1/4 ?
            patience=1,  # xxx 0 or 1 or 2 ?
            threshold=1.0e-2,
            min_lr=1.0e-2,  # xxx value?
            verbose=True)
        return [optimizer]  #, [lr_sched]

    def train_dataloader(self):
        return DataLoader(
            config.data_train,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=morph_batcher,
            #num_workers=1
        )

    def val_dataloader(self):
        return DataLoader(
            config.data_val,
            batch_size=len(config.data_val),
            shuffle=False,
            collate_fn=morph_batcher,
            #num_workers=1
        )

    # todo: add me
    #def test_dataloader(self):
    #    return DataLoader(
    #        config.data_train, # xxx data_train
    #        batch_size = config.data_train.ndata,
    #        shuffle = False,
    #        collate_fn = morph_batcher)

    def loss_func(self, pred, output):
        # Decode left-to-right
        logprobs = config.decoder(pred.form)
        losses = self.criterion(logprobs, output.form_id)
        # Sum losses over positions, average over batch xxx document
        loss = torch.sum(losses, dim=1)
        loss = torch.mean(loss)

        # L1 regularization of phonological contexts
        l1_losses = []
        l1_lambda = 1.0e-4  # xxx config option
        if self.cogrammar.phonology is not None:
            for mod in self.cogrammar.phonology.matcher.modules():
                if isinstance(mod, LiteralMatcher):
                    l1_losses.append(torch.norm(mod.W.weight, 1))
                    l1_losses.append(torch.norm(mod.W.bias, 1))
            #print(f'l1_losses {l1_losses}')
            loss += l1_lambda * torch.mean(torch.tensor(l1_losses))

        return loss, losses

    # # # # # Deprecated # # # # #

    def risk_loss(self, pred, output):
        # Decode predictions to strings xxx use form_embedder instead
        preds = config.decoder.decode2string(pred.form)
        #print(f'target outputs: {output.form_str}')
        #print(f'predicted outputs: {preds}')
        # Normalized edit distance (Makarov & Clematide 2018)
        costs = [
            editdistance.eval(x, y) for (x, y) in zip(preds, output.form_str)
        ]
        max_lens = [
            max(len(x.split()), len(y.split()))
            for (x, y) in zip(preds, output.form_str)
        ]
        #print(costs)
        #print(max_lens)
        costs = torch.FloatTensor(costs) / \
                torch.FloatTensor(max_lens)
        #print(f'normalized edit distances: {costs}')
        costs = costs - torch.FloatTensor([x==y \
            for (x,y) in zip(preds, output.form_str)])
        # Approximate risk
        log_prob = config.log_prob
        q = exp(log_prob)
        q = q / torch.sum(q, 0)
        #print(f'q: {q}, sum(q) = {torch.sum(q,0)}')
        losses = q * costs
        loss = torch.sum(losses, 0)
        #print(f'losses: {losses}')
        #sys.exit(0)
        #print(f'loss: {loss}')
        #sys.exit(0)
        return loss, losses