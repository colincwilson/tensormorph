#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr, radial_basis
from tpr import *
from radial_basis import GaussianPool

# soft regex match over window of length 3
class Matcher3(nn.Module):
    def __init__(self, morpho_size, input_size, sim='dot'):
        super(Matcher3, self).__init__()
        self.matcher_prev = Matcher(morpho_size, input_size)
        self.matcher_i    = Matcher(morpho_size, input_size)
        self.matcher_next = Matcher(morpho_size, input_size)
    
    def forward(self, X, morpho):
        nbatch, m, n = X.shape
        zero = torch.zeros((nbatch, m, 1))
        _X_ = torch.cat((zero, X, zero), 2)
        log_match_prev  = self.matcher_prev(_X_, morpho).narrow(1,0,n)
        log_match_i     = self.matcher_i(_X_, morpho).narrow(1,1,n)
        log_match_next  = self.matcher_next(_X_, morpho).narrow(1,2,n)
        match = torch.exp(log_match_prev + log_match_i + log_match_next)
        mask = hardtanh(X.narrow(1,0,1), 0.0, 1.0).squeeze(1) # detach?
        match = match * mask
        assert(np.all(match.data.numpy() >= 0.0))
        return match


# soft regex match of a single filler
class Matcher(nn.Module):
    def __init__(self, morpho_size, nfeature):
        super(Matcher, self).__init__()
        self.morph2w    = nn.Linear(morpho_size, nfeature, bias=True)
        self.morph2b    = nn.Linear(morpho_size, 1, bias=True)
        self.morph2tau  = nn.Linear(morpho_size, 1, bias=True)
        self.nfeature = nfeature

    def forward(self, X, morpho):
        w = self.morph2w(morpho).unsqueeze(1)
        b = self.morph2b(morpho).unsqueeze(1)
        tau = self.morph2tau(morpho).unsqueeze(1)
        k = self.nfeature
        if tpr.random_roles:
            # distributed roles -> local roles
            X = torch.bmm(X, tpr.U)
        # log_match_i = tau * dot(w,x_i)
        score     = torch.bmm(w, X.narrow(1,0,k)) + b
        log_match = logsigmoid(tau * score).squeeze(1)
        # log_match_i = dot(w,x_i) - ||w||_1 
        #log_match = torch.bmm(w, X.narrow(1,0,k))
        #log_match = log_match.squeeze(1) - torch.norm(w, 1, 2)
        return log_match

# old code w/ alternative similarity functions
# def forward(self, X, morpho):
#     w   = self.morph2w(morpho)
#     b   = self.morph2b(morpho)
#     tau = torch.exp(self.morph2tau(morpho))
#     k   = self.input_size

#     # calculate log match for each filler in each batch
#     if self.sim=='dot': # scaled sigmoid of (dot product + bias)
#         w, b, tau = w.unsqueeze(1), b.unsqueeze(1), tau.unsqueeze(1)
#         score = torch.bmm(w, X.narrow(1,0,k)) + b
#         log_match = logsigmoid(tau * score).squeeze(1)
#         return log_match
#     if self.sim=='alcove': # scaled gaussian rbf (Kruschke)
#         w     = tanh(w).unsqueeze(-1)
#         delta = X.narrow(1,0,k) - w
#         score = torch.pow(delta, 2.0)
#         score = torch.bmm(score.transpose(1,2), torch.abs(w))
#         score = torch.pow(score, 0.5).squeeze(-1)
#         log_match = -tau * score
#         #print 'w', w.data[0,:,0].numpy()
#         #print log_match.data[0,:].numpy()
#         return log_match
#     if self.sim=='smo86': # harmony function of Smolensky 1986
#         w = tanh(w)
#         score = torch.bmm(w.unsqueeze(1), X.narrow(1,0,k))
#         w_norm = torch.sum(torch.abs(w), 1, keepdim=True)
#         score = (score / w_norm.unsqueeze(-1)).squeeze(1)
#         score = tau * (score - 1.0)
#         return score
#     return None
