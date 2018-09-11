#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr, radial_basis
from tpr import *
from radial_basis import GaussianPool

# soft regex match over window of length 3
class Matcher3(nn.Module):
    def __init__(self, morpho_size, input_size, node=''):
        super(Matcher3, self).__init__()
        self.matcher_prev = MatcherGCM(morpho_size, input_size, node=node+'-prev')
        self.matcher_cntr = MatcherGCM(morpho_size, input_size, node=node+'-cntr')
        self.matcher_next = MatcherGCM(morpho_size, input_size, node=node+'-next')
        self.node = node
    
    def forward(self, X, morpho):
        nbatch, m, n = X.shape
        # pad input tpr on both sides with epsilon filler
        zero = torch.zeros((nbatch, m, 1))
        _X_ = torch.cat((zero, X, zero), 2)
        # apply matchers to every window of length three in input
        log_match_prev = self.matcher_prev(_X_, morpho).narrow(1,0,n)
        log_match_cntr = self.matcher_cntr(_X_, morpho).narrow(1,1,n)
        log_match_next = self.matcher_next(_X_, morpho).narrow(1,2,n)
        # multiplicative (log-linear) combination of matcher outputs
        match = torch.exp(log_match_prev + log_match_cntr + log_match_next)
        # mask out match results for epsilon fillers
        mask = hardtanh(X.narrow(1,0,1), 0.0, 1.0).squeeze(1) # detach?
        match = match * mask
        assert(np.all(match.data.numpy() >= 0.0))

        if tpr.recorder is not None:
            tpr.recorder.set_values(self.node, {
                'match':match
            })

        return match


# soft regex match of a single filler
class Matcher(nn.Module):
    def __init__(self, morpho_size, nfeature, node=''):
        super(Matcher, self).__init__()
        self.morph2w    = nn.Linear(morpho_size, nfeature, bias=True)
        self.morph2b    = nn.Linear(morpho_size, 1, bias=True)
        self.morph2tau  = nn.Linear(morpho_size, 1, bias=True)
        self.nfeature = nfeature
        self.node = node

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


# matcher with attention weights as in GCM (Nosofsky 1986), 
# ALCOVE (Kruschke 1991), etc. after Shepard (1962, 1987)
class MatcherGCM(nn.Module):
    def __init__(self, morpho_size, nfeature, node=''):
        super(MatcherGCM, self).__init__()
        self.morph2a = nn.Linear(morpho_size, nfeature, bias=True)
        self.morph2w = nn.Linear(morpho_size, nfeature, bias=True)
        self.morph2c = nn.Linear(morpho_size, 1, bias=True)
        self.nfeature = nfeature
        self.node = node
    
    def forward(self, X, morpho):
        # attention weights in [0,1]
        a = sigmoid(self.morph2a(morpho)).unsqueeze(2)
        # specified values in [-1,+1]
        w = tanh(self.morph2w(morpho)).unsqueeze(2)
        # sensitivity > 0
        c = torch.exp(self.morph2c(morpho))
        k = self.nfeature

        if tpr.random_roles:
            # distributed roles -> local roles
            X = torch.bmm(X, tpr.U)
        #score = torch.abs(X.narrow(1,0,k) - w)
        score = torch.pow(X.narrow(1,0,k) - w, 2.0)
        score = torch.sum(a * score, 1)
        score = torch.pow(score, 0.5)
        log_match = -c * score

        if tpr.recorder is not None:
            tpr.recorder.set_values(self.node, {
                'a':a, 'w':w, 'c':c, 
                'score':score, 
                'log_match':log_match
            })

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
#     if self.sim=='harmonium': # harmony function of Smolensky 1986
#         w = tanh(w)
#         score = torch.bmm(w.unsqueeze(1), X.narrow(1,0,k))
#         w_norm = torch.sum(torch.abs(w), 1, keepdim=True)
#         score = (score / w_norm.unsqueeze(-1)).squeeze(1)
#         score = tau * (score - 1.0)
#         return score
#     return None
