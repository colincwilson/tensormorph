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
        for W_ix in ['W_ir', 'W_iz', 'W_in']:
            setattr(self, W_ix, nn.Linear(input_size, hidden_size))
        for W_hx in ['W_hr', 'W_hz', 'W_hn']:
            setattr(self, W_hx, nn.Linear(hidden_size, hidden_size))
        # xxx weight initialization?
    
    def forward(self, input, input_len=None):
        max_len, batch_size, _ = input.shape
        h_init = torch.zeros(batch_size, self.hidden_size, requires_grad=False)
        R, Z, N, H = [], [], [], []
        for t in range(max_len):
            x_t = input[t,:,:]
            h_prev = h_init if t==0 else H[-1]
            r_t = torch.sigmoid(self.W_ir(x_t) + self.W_hr(h_prev))
            z_t = torch.sigmoid(self.W_iz(x_t) + self.W_hz(h_prev))
            n_t = torch.tanh(self.W_in(x_t) + r_t * self.W_hn(h_prev))
            h_t = (1.0 - z_t) * n_t + z_t * h_prev
            R.append(r_t); Z.append(z_t); N.append(n_t); H.append(h_t)

        for name,result in zip(['R','Z','N','H'], [R,Z,N,H]):
            setattr(self, name, torch.stack(result,0))
        if input_len is not None:
            self.mask_results(input_len)
            return self.H[input_len-1, torch.arange(batch_size)]
        return self.H.unsqueeze(0)

    def mask_results(self, input_len):
        batch_size = len(input_len) #input_len.shape[0]
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
    inpt_len = torch.as_tensor([2,1,3,1])
    H = gru1(inpt, inpt_len) # H is batch_size x hidden_size
    print (H)
    print (gru1.Z)

if __name__ == '__main__':
    main()