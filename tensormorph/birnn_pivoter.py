# -*- coding: utf-8 -*-

import config
from tpr import *
from context_rnn import *
#import distributions as distrib
#import mcmc


class BiRNNPivoter(nn.Module):
    """
    Pivot function implemented with generic bidirectional RNN
    """

    def __init__(self, nfeature=3, nhidden=3, use_context=True):
        super(BiRNNPivoter, self).__init__()
        self.nfeature = nfeature
        self.nhidden = nhidden
        self.use_context = use_context
        dcontext = config.dcontext
        if use_context:  # GRU modulated by context
            self.rnn_lr = ContextGRU(
                input_size=nfeature,
                hidden_size=nhidden,
                context_size=dcontext,
                direction='LR->')
            self.rnn_rl = ContextGRU(
                input_size=nfeature,
                hidden_size=nhidden,
                context_size=dcontext,
                direction='<-RL')
        else:  # non-contextual GRU xxx add epsilon-masking
            self.rnn = nn.GRU(
                input_size=nfeature,
                hidden_size=nhidden,
                batch_first=True,
                bidirectional=True)
        self.linear = torch.nn.Linear(2 * nhidden, 1)

    def forward(self, base, context):
        # Prepare form for scanning
        form = distrib2local(base.form)
        nbatch = form.shape[0]
        mask = exp(epsilon_mask(form))
        form = form.transpose(1, 2).narrow(-1, 0, self.nfeature)

        # Bidirectional scan
        if self.use_context:
            h_lr = self.rnn_lr(form, mask, context)
            h_rl = self.rnn_rl(form, mask, context)
            h = torch.cat([h_lr, h_rl], 1).transpose(2, 1)
        else:
            h, _ = self.rnn(form)
        #print(h_lr[0])

        # Linear map and masking
        pivot_logit = self.linear(h).squeeze(-1)
        pivot = mask * torch.sigmoid(pivot_logit)
        return pivot
