#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

# Single-layer, unidirectional GRU with no dropout
class GRU1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, bidirectional=False):
        super(GRU1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers    # ignored
        self.bias = bias    # ignored
        self.bidirectional = bidirectional  # ignored
        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_iz = nn.Linear(input_size, hidden_size)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, input, input_len=None):
        max_len, batch_size, _ = input.shape
        h_init = torch.zeros(batch_size, self.hidden_size, requires_grad=False)
        R, Z, N, H = [], [], [], []
        for t in range(max_len):
            xt = input[t,:,:]
            h_prev = h_init if t==0 else H[-1]
            rt = torch.sigmoid(self.W_ir(xt) + self.W_hr(h_prev))
            zt = torch.sigmoid(self.W_iz(xt) + self.W_hz(h_prev))
            nt = torch.tanh(self.W_in(xt) + rt * self.W_hn(h_prev))
            ht = (1.0 - zt) * nt + zt * h_prev
            R.append(rt); Z.append(zt); N.append(nt); H.append(ht)

        for name,result in zip(['R','Z','N','H'], [R,Z,N,H]):
            setattr(self, name, torch.stack(result,0))
        if input_len is not None:
            self.mask_results(input_len)
            print (self.H.shape, input_len.shape)
            print (input_len)
            return self.H[input_len-1, torch.arange(self.H.shape[0])]
        return self.H

    def mask_results(self, input_len):
        batch_size = input_len.shape[0]
        mask = torch.zeros_like(self.H).byte()
        for i in range(batch_size):
            mask[:input_len[i],i,:] = 1
        for name in ['R', 'Z', 'N', 'H']:
            result = getattr(self, name)
            result.masked_fill_(1-mask, 0.0)

def main():
    max_len = 3
    batch_size = 4
    input_size = 5
    hidden_size = 6
    gru1 = GRU1(input_size, hidden_size)

    inpt = torch.rand(max_len, batch_size, input_size)
    inpt_len = torch.as_tensor([1,3,2])
    H = gru1(inpt, inpt_len)
    print (H.shape)
    sys.exit(0)
    for i in range(batch_size):
        print (H[:,i,:])
        print (gru1.Z[:,i,:])

if __name__ == '__main__':
    main()