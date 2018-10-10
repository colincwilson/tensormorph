#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tpr, radial_basis
from tpr import *
from radial_basis import GaussianPool

# soft regex match over window of length 3
class Matcher3(nn.Module):
    def __init__(self, morpho_size, input_size, node=''):
        super(Matcher3, self).__init__()
        self.matcher_prev = Matcher(morpho_size, input_size, node=node+'-prev')
        self.matcher_cntr = Matcher(morpho_size, input_size, node=node+'-cntr')
        self.matcher_next = Matcher(morpho_size, input_size, node=node+'-next')
        self.node = node
    
    def forward(self, X, morpho):
        nbatch, m, n = X.shape
        # pad input tpr on both sides with epsilon filler
        zero = torch.zeros((nbatch, m, 1))
        _X_ = torch.cat((zero, X, zero), 2)
        # apply matchers to every window of length three in input
        match_prev = self.matcher_prev(_X_, morpho).narrow(1,0,n)
        match_cntr = self.matcher_cntr(_X_, morpho).narrow(1,1,n)
        match_next = self.matcher_next(_X_, morpho).narrow(1,2,n)
        # multiplicative combination of matcher outputs
        match = match_prev * match_cntr * match_next
        # mask out match results for epsilon fillers
        mask = hardtanh(X.narrow(1,0,1), 0.0, 1.0).squeeze(1).detach()
        match = match * mask
        try:
            assert(np.all(0.0 <= match.data.numpy()))
            assert(np.all(match.data.numpy() <= 1.0))
        except AssertionError as e:
            print(match.data.numpy())
            raise

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
        self.nfeature = nfeature
        self.node = node

    def forward(self, X, morpho):
        w   = sigmoid(self.morph2w(morpho).unsqueeze(1))
        w_b = w + (relu(torch.sign(w - 0.5)) - w).detach()
        #print(np.round(w_b.data[0,:].numpy(), 2))
        k = self.nfeature
        if tpr.random_roles:
            # distributed roles -> local roles
            X = torch.bmm(X, tpr.U)
        # straight-through
        match = torch.bmm(w_b, X.narrow(1,0,k)).squeeze(1) - torch.sum(w_b,2) + 0.5
        #print(match.shape)
        match_b = match + (relu(torch.sign(match)) - match).detach()
        print(np.round(w_b.data[0,:].numpy(), 2))
        print(np.round(match.data[0,:].numpy(), 2))
        return match_b


# attention-weighted euclidean distance matcher  
# see GCM (Nosofsky 1986), ALCOVE (Kruschke 1991), etc. 
# after Shepard (1962, 1987)
class MatcherGCM(nn.Module):
    def __init__(self, morpho_size, nfeature, node=''):
        super(MatcherGCM, self).__init__()
        self.morph2w = nn.Linear(morpho_size, nfeature, bias=True)
        self.morph2a = nn.Linear(morpho_size, nfeature, bias=True)
        self.morph2c = nn.Linear(morpho_size, 1, bias=True)
        self.nfeature = nfeature
        self.node = node
    
    def forward(self, X, morpho):
        # feature specifications in [-1,+1]
        w = tanh(self.morph2w(morpho)).unsqueeze(2)
        # attention weights in [0,1]
        a = sigmoid(self.morph2a(morpho)).unsqueeze(2)
        #a = torch.abs(w)
        #a = torch.pow(w, 2.0)
        # sensitivity > 0
        c = torch.exp(self.morph2c(morpho))
        k = self.nfeature

        if tpr.random_roles:
            # distributed roles -> local roles
            X = torch.bmm(X, tpr.U)
        #print X.narrow(1,0,k).shape, w.shape, a.shape
        #print w[0,:,:]
        #print a[0,:,:]
        score = torch.pow(X.narrow(1,0,k) - w, 2.0)
        #score = torch.pow(X.narrow(1,1,k) - w, 2.0)
        score = torch.sum(a * score, 1)
        score = torch.pow(score, 0.5)
        log_match = -c * score
        #print log_match[0,:]
        #print torch.exp(log_match[0,:])
        #sys.exit(0)

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
