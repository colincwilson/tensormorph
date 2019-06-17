#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
from .tpr import *
from .radial_basis import GaussianPool
from .distance import pairwise_distance

# soft regex match over window of length 3
class Matcher3(nn.Module):
    """
    Soft regex match over window of length 3
    if npattern == 1, returns matches to a single soft regex in the form (nbatch, nrole)
    else (npattern>1) returns
    - all matches to a bank of soft regexes in the form (nbatch, nrole, npattern) if maxout==False
    - the result of maxout (nbatch, nrole, npattern) -> (nbatch, nrole) if maxout==True
    """
    def __init__(self, morpho_size, nfeature, npattern=1, maxout=False, node=''):
        super(Matcher3, self).__init__()
        self.matcher_prev   = MatcherGCM(morpho_size, nfeature, npattern, node=node+'-prev')
        self.matcher_cntr   = MatcherGCM(morpho_size, nfeature, npattern, node=node+'-cntr')
        self.matcher_next   = MatcherGCM(morpho_size, nfeature, npattern, node=node+'-next')
        self.npattern       = npattern
        self.maxout         = maxout
        self.node           = node
    
    def forward(self, X, morpho):
        nbatch, m, n = X.shape
        # Pad input tpr on both sides with epsilon filler
        zero = torch.zeros((nbatch, m, 1))
        _X_ = torch.cat((zero, X, zero), 2)

        # Apply matchers to every window of length three in input
        log_match_prev = self.matcher_prev(_X_, morpho).narrow(1,0,n)
        log_match_cntr = self.matcher_cntr(_X_, morpho).narrow(1,1,n)
        log_match_next = self.matcher_next(_X_, morpho).narrow(1,2,n)

        # Multiplicative (log-linear) combination of matcher outputs
        matches = torch.exp(log_match_prev + log_match_cntr + log_match_next)
        if config.discretize:
            matches = torch.round(matches)

        # Mask out match results for epsilon fillers
        mask = hardtanh(X.narrow(1,0,1), 0.0, 1.0)\
                .squeeze(1).unsqueeze(-1).detach()
        matches = matches * mask
        try:
            assert(np.all(0.0 <= matches.data.numpy()))
            assert(np.all(matches.data.numpy() <= 1.0))
        except AssertionError as e:
            print(matches.data.numpy())
            raise

        # Reduce output of matches (with squeeze or maxout)
        if self.npattern==1:
            match = matches.squeeze(-1)
        elif self.maxout:
            match,_ = torch.max(matches, 2)

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'matches':matches,
                'match':match
            })

        return match


# xxx deprecate?
class Matcher(nn.Module):
    """
    Soft regex match of a single filler
    """
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
        if config.random_roles:
            # distributed roles -> local roles
            X = torch.bmm(X, config.U)
        # log_match_i = tau * dot(w,x_i)
        score     = torch.bmm(w, X.narrow(1,0,k)) + b
        log_match = logsigmoid(tau * score).squeeze(1)
        return log_match


class MatcherGCM(nn.Module):
    """
    Attention-weighted euclidean distance matcher,
    see GCM (Nosofsky 1986), ALCOVE (Kruschke 1991), etc. after Shepard (1962, 1987)
    if npattern==1, returns a single match in the form (nbatch, nrole)
    else (npattern>1) returns a bank of matches in the form (nbatch, nrole, npattern)
    """
    def __init__(self, morpho_size, nfeature, npattern=1, node=''):
        super(MatcherGCM, self).__init__()
        self.morph2W    = nn.Linear(morpho_size, nfeature*npattern, bias=True)
        self.morph2A    = nn.Linear(morpho_size, nfeature*npattern, bias=True)
        self.morph2c    = nn.Linear(morpho_size, npattern, bias=True)
        self.nfeature   = nfeature
        self.npattern   = npattern
        self.node       = node
    
    def forward(self, X, morpho):
        nbatch = X.shape[0]
        nfeature, npattern = self.nfeature, self.npattern
        # Feature specifications in [-1,+1] xxx see config.privative_ftrs
        W = tanh(self.morph2W(morpho))\
            .view(nbatch, nfeature, npattern)
        #w = tanh(self.morph2w(morpho)).unsqueeze(2)
        # Attention weights in [0,1]
        A = sigmoid(self.morph2A(morpho))\
            .view(nbatch, nfeature, npattern)
        #a = sigmoid(self.morph2a(morpho)).unsqueeze(2)
        #a = torch.abs(w)
        # Sensitivity > 0
        c = torch.exp(self.morph2c(morpho)).unsqueeze(1)
        k = self.nfeature
        if config.discretize:
            W = torch.round(W)
            A = torch.round(A)

        if config.random_roles:
            # distributed roles -> local roles
            X = torch.bmm(X, config.U)

        score = pairwise_distance(X.narrow(1,0,k), W, A)
        log_match = -c * score
        #score = torch.pow(X.narrow(1,0,k) - w, 2.0)
        #score = torch.sum(a * score, 1)
        #score = torch.pow(score, 0.5)
        #log_match = -c * score

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'a':A, 'w':W, 'c':c, 
                'score':score, 
                'log_match':log_match
            })

        return log_match


# alternative similarity functions
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
