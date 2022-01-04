# -*- coding: utf-8 -*-

import config
from tpr import *
from torch.nn import LSTMCell
#from pytorch_lightning import Trainer
from data_util import morph_batcher
from morph import Morph
from random import shuffle


class SequenceGenerator(nn.Module):
    """
    Sequence generator, backed by recurrent network, that 
    updates its state only when attended
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super(SequenceGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnns = nn.ModuleList()
        for l in range(num_layers):
            rnn = LSTMCell(input_size=(input_size if l == 0 else hidden_size),
                           hidden_size=hidden_size,
                           bias=bias)
            self.rnns.append(rnn)

    def reset(self, nbatch):
        """
        Reset RNN state
        """
        self.h = [
            torch.zeros(nbatch, self.hidden_size)
            for i in range(self.num_layers)
        ]
        self.c = [
            torch.zeros(nbatch, self.hidden_size)
            for i in range(self.num_layers)
        ]

    def forward(self, x, alpha):
        """
        Generate sequence member from input, 
        advance RNN state if attended
        """
        h, c = self.h, self.c
        y = x
        for l in range(self.num_layers):
            h_new, c_new = self.rnns[l](y, (h[l], c[l]))
            h[l] = alpha * h_new + (1.0 - alpha) * h[l]
            c[l] = alpha * c_new + (1.0 - alpha) * c[l]
            y = h_new
        return y


def train():
    # Sequence generator
    input_size = 50
    seqgen = SequenceGenerator(input_size=input_size, hidden_size=50)

    learning_rate = 0.1
    optimizer = torch.optim.Adagrad(seqgen.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    # Assign arbitray code to each example
    for ex in config.dat_train:
        ex['code'] = torch.randn(input_size)

    # Map code to predicted output
    nbatch = 48
    for r in range(2000):
        shuffle(config.dat_train.dat_embed)
        batch = config.dat_train.dat_embed[:nbatch]
        x = [ex['code'] for ex in batch]
        x = torch.stack(x, dim=0)
        output = [ex['output'] for ex in batch]
        output = torch.stack(output, dim=0)
        output_str = [ex['output_str'] for ex in batch]
        alpha = torch.ones(nbatch, 1)

        seqgen.reset(nbatch)
        ys = []
        for t in range(config.nrole):
            y = seqgen(x, alpha)
            ys.append(y)
        y = torch.stack(ys, -1)
        y = y[:, :config.dsym, :]

        pred = config.decoder(y)
        optimizer.zero_grad()
        losses = criterion(pred, output)
        loss = torch.sum(losses, dim=1)
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        if r % 50 == 0:
            print(loss)
            print(output_str)
            print(Morph(y)._str())

    #ex = batch[0]
    #x = ex['code'].unsqueeze(0)
    #alpha = torch.ones(nbatch, 1)


def test():
    input_size = 4
    hidden_size = 4
    nbatch = 1
    seq_len = 5
    rnn = SequenceGenerator(input_size, hidden_size, num_layers=1)
    rnn.reset(nbatch)

    x = torch.randn(nbatch, input_size)
    x = x.unsqueeze(1).expand(nbatch, seq_len, input_size)
    alpha = torch.ones(nbatch, seq_len, 1)
    alpha.data[:, 1, 0] = 0.01
    #alpha.data[:,6:7,0] = 0.01

    ys = []
    for i in range(seq_len):
        y = rnn(x[:, i], alpha[:, i])
        ys.append(y)
    y = torch.stack(ys, -1)
    print(y[0])


if __name__ == "__main__":
    test()
