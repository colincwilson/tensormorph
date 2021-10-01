# -*- coding: utf-8 -*-
# RNN parameterized by linear regression on context

import config
from tpr import *


class ContextGRU(nn.Module):

    def __init__(self, input_size, hidden_size, context_size, direction='LR->'):
        super(ContextGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.direction = direction
        self.context2weight_ih = nn.Linear(context_size,
                                           3 * input_size * hidden_size)
        self.context2weight_hh = nn.Linear(context_size,
                                           3 * hidden_size * hidden_size)
        self.context2bias_ih = nn.Linear(context_size, hidden_size * 3)
        self.context2bias_hh = nn.Linear(context_size, hidden_size * 3)
        if context_size == 1:
            self.context2weight_ih.weight.detach()
            self.context2weight_hh.weight.detach()
            self.context2bias_ih.weight.detach()
            self.context2bias_hh.weight.detach()

    def forward(self, form, mask, context):
        nbatch = form.shape[0]
        # Map context to GRU parameters
        W_ir, W_iz, W_in = \
            self.context2weight_ih(context) \
            .view(nbatch, self.input_size, -1) \
            .chunk(3, dim = 2)
        W_hr, W_hz, W_hn = \
            self.context2weight_hh(context) \
            .view(nbatch, self.hidden_size, -1) \
            .chunk(3, dim = 2)
        b_ir, b_iz, b_in = \
            self.context2bias_ih(context) \
            .view(nbatch, -1).chunk(3, dim = 1)
        b_hr, b_hz, b_hn = \
            self.context2bias_hh(context) \
            .view(nbatch, -1).chunk(3, dim = 1)

        # Scan in designated direction
        if self.direction == 'LR->':
            begin, end, step = 0, config.nrole, 1
        else:
            begin, end, step = config.nrole - 1, -1, -1
        h = torch.zeros(nbatch, self.hidden_size)  # initialize hidden
        H = [None] * config.nrole  # list of hiddens after initial
        for t in range(begin, end, step):
            x = form[:, t]
            m = mask[:, t].unsqueeze(-1)
            r = sigmoid(bmatvec(W_ir, x) + b_ir + bmatvec(W_hr, h) + b_hr)
            z = sigmoid(bmatvec(W_iz, x) + b_iz + bmatvec(W_hz, h) + b_hz)
            n = tanh(bmatvec(W_in, x) + b_in + r * (bmatvec(W_hn, h) + b_hn))
            h = m * ((1 - z) * n + z * h)  # + (1 - m) * 0
            H[t] = h

        # Return hidden state sequence
        h = torch.stack(H, -1)
        return h
