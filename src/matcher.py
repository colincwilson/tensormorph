#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .environ import config
from .tpr import *
from .radial_basis import GaussianPool
from .distance import pairwise_distance
from .phon_features import ftrspec2vec

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
        self.matcher_prev   = MatcherWinnow(morpho_size, nfeature, npattern, node=node+'-prev')
        self.matcher_cntr   = MatcherWinnow(morpho_size, nfeature, npattern, node=node+'-cntr')
        self.matcher_next   = MatcherWinnow(morpho_size, nfeature, npattern, node=node+'-next')
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

    def compile(self, ftrspecs, c=1.0):
        """
        Compile instance from feature specification, 
        one for each matcher
        """
        for i,matcher in enumerate(['matcher_prev', 'matcher_cntr', 'matcher_next']):
            matcher = getattr(self, matcher)
            matcher.compile(ftrspecs[i], c)


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
    see GCM (Nosofsky 1986), ALCOVE (Kruschke 1991), etc. after Shepard (1962, 1987).
    If npattern==1, returns a single match with shape (nbatch, nrole)
    else (npattern>1) returns a bank of matches with shape (nbatch, nrole, npattern)
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
        # Feature specifications in [-1,+1] xxx check config.privative_ftrs
        W = tanh(self.morph2W(morpho))\
            .view(nbatch, nfeature, npattern)
            
        # Attention weights in [0,1]
        A = sigmoid(self.morph2A(morpho))\
            .view(nbatch, nfeature, npattern)

        # Sensitivity > 0
        c = torch.exp(self.morph2c(morpho)).unsqueeze(1)
        k = self.nfeature
        if config.discretize:
            W = torch.round(W)
            A = torch.round(A)

        if config.random_roles:
            # distributed roles -> local roles
            X = torch.bmm(X, config.U)

        # Match of each filler determined by 
        # attention-weighted euclidean distance
        score = pairwise_distance(X.narrow(1,0,k), W, A)
        log_match = -c * score

        # Mask out matches to epsilon, in log domain
        mask = hardtanh(X.narrow(1,0,1), 0.0, 1.0)
        log_mask = torch.log(mask).transpose(1,2)
        log_match += log_mask

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'a':A, 'w':W, 'c':c, 
                'score':score, 
                'log_match':log_match
            })

        return log_match
    

    def compile(self, ftrspec, c=1.0):
        """
        Compile instance from feature specifications 
        (see ftrspec2vec) and specificity
        """
        w, a = ftrspec2vec(ftrspec)
        w = torch.FloatTensor(w)
        a = torch.FloatTensor(a)
        # note: weight specs correct for morph2X nonlinearities
        self.morph2W.weight.data[:,0] = 10.0*w
        self.morph2A.weight.data[:,0] = 20.0*(a - 0.5)
        self.morph2c.weight.data[:] = c
        self.morph2W.bias.data[:] = 0.0
        self.morph2A.bias.data[:] = 0.0
        self.morph2c.bias.data[:] = 0.0


class MatcherWinnow(nn.Module):
    """
    Weighted squared distance matcher inspired by classic models 
    for learning conjunctions of boolean literals, such as Winnow.
    Assumes that each feature is binary (could be modified to accept 
    hard-coded set of values for each feature) with possibility of underspecification.
    If npattern == 1, returns a single match with shape (nbatch, nrole)
    else (npattern > 1) returns a bank of matches with shape (nbatch, nrole, npattern)
    # xxx npattern > 1 broken
    """
    def __init__(self, morpho_size, nfeature, npattern=1, normalize=True, node=''):
        super(MatcherWinnow, self).__init__()
        self.morph2Wplus = nn.Linear(morpho_size, nfeature*npattern, bias=True)
        self.morph2Wminus = nn.Linear(morpho_size, nfeature*npattern, bias=True)
        self.nfeature = nfeature
        self.npattern = npattern
        self.normalize = normalize
        self.node = node
    
    def forward(self, X, morpho):
        nbatch = X.shape[0]
        nfeature, npattern = self.nfeature, self.npattern

        # Weights on positive (+1) and negative (-1) specifications
        # xxx clamp values to zero according to config.privative_ftrs
        # xxx replace exp with ReLU or other all-positive function?
        Wplus = torch.exp(self.morph2Wplus(morpho)) \
                .view(nbatch, nfeature, npattern)
        Wminus = torch.exp(self.morph2Wminus(morpho)) \
                .view(nbatch, nfeature, npattern)

        # Normalize weights (with weight of underspecification fixed at unity)
        if self.normalize:
            Z = Wplus + Wminus + 1.0
            Wplus = Wplus / Z
            Wminus = Wplus / Z

        if config.random_roles:
            # distributed roles -> local roles
            X = torch.bmm(X, config.U)

        # Match of each filler determined by squared distance 
        # to ideal +/-1 values
        dist = Wplus * (X.narrow(1,0,nfeature) - 1.0)**2.0 +\
               Wminus * (X.narrow(1,0,nfeature) + 1.0)**2.0
        dist = torch.sum(dist, 1)
        log_match = -dist

        # Mask out matches to epsilon, in log domain
        mask = hardtanh(X.narrow(1,0,1), 0.0, 1.0)
        log_mask = torch.log(mask).squeeze(1)
        log_match += log_mask

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'Wplus':Wplus, 'Wminus':Wminus, 
                'dist':dist,
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
