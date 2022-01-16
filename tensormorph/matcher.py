# -*- coding: utf-8 -*-

import config
from tpr import *
from scanner import propagate
from radial_basis import GaussianPool
from distance import pairwise_distance
#from phon_features import ftrspec2vec
from torch.nn.functional import softplus


class Matcher3(nn.Module):
    """
    Length-3 pattern(s) of gradient feature matrices. If npattern == 1, return matches to a single soft regex in the form [nbatch x nrole]. Otherwise (npattern > 1), returns all matches to a bank of soft regexes in the form [nbatch x nrole x npattern] if maxout is False, else appplies maxout to reduce output to [nbatch x nrole]
    """

    def __init__(self, dcontext, nfeature, npattern=1, maxout=False):
        super(Matcher3, self).__init__()
        self.matcher_prev = \
            LiteralMatcher(dcontext, nfeature, npattern)
        self.matcher_cntr = \
            LiteralMatcher(dcontext, nfeature, npattern)
        self.matcher_next = \
            LiteralMatcher(dcontext, nfeature, npattern)
        self.npattern = npattern
        self.maxout = maxout

    def forward(self, X, context):
        """
        Gradient match to feature pattern(s) at all positions in X, computed in pre-tanh domain. Returns log match values, apply exp() to output for values in (0,1)
        """
        nbatch, m, n = X.shape
        self.trace = {}

        # Pad input form on both sides with epsilon filler
        zero = torch.zeros((nbatch, m, 1), device=config.device)
        _X_ = torch.cat((zero, X, zero), 2)

        # Apply matchers to every window of length three in input form
        match_prev, match_cntr, match_next = \
            self.matcher_prev(_X_, context).narrow(2,0,n), \
            self.matcher_cntr(_X_, context).narrow(2,1,n), \
            self.matcher_next(_X_, context).narrow(2,2,n)

        # Log-linear combination of matcher outputs
        matches = match_prev + match_cntr + match_next

        # Reduce output of matches (with squeeze or maxout)
        if self.npattern == 1:
            match = matches.squeeze(1)
        elif self.maxout:
            match, _ = torch.max(matches, -1)
        else:
            match = matches

        self.trace['matches'] = matches.clone().detach()
        self.trace['match'] = match.clone().detach()
        return match

    # xxx broken
    def compile(self, ftrspecs, c=1.0):
        """
        Compile instance from feature specification, 
        one for each matcher
        """
        for i, matcher in enumerate(
            ['matcher_prev', 'matcher_cntr', 'matcher_next']):
            matcher = getattr(self, matcher)
            matcher.compile(ftrspecs[i], c)

    def print(self, context):
        return {
            'prev': self.matcher_prev.print(context),
            'cntr': self.matcher_cntr.print(context),
            'next': self.matcher_next.print(context)
        }


class LiteralMatcher(nn.Module):
    """
    Phonological patterns defined over equipollent features; each pattern is a gradient approximation of a conjunction of boolean literals (+F or -F)
    """

    def __init__(self, dcontext, nfeature, npattern=1, normalize=True):
        super(LiteralMatcher, self).__init__()
        self.npattern = npattern
        self.nfeature = nfeature
        self.normalize = normalize
        self.W = nn.Linear(dcontext, npattern * nfeature, bias=True)
        #torch.nn.init.normal_(self.W.weight, 0.0, 0.1)
        #torch.nn.init.normal_(self.W.bias, 0.0, 0.1)

    def forward(self, X, context):
        """
        Gradient match to feature pattern(s) at all positions in form X, computed in pre-tanh domain. Returns log match, apply exp() to output for values in (0,1)
        """
        nbatch = X.shape[0]
        npattern = self.npattern
        nfeature = self.nfeature
        self.trace = {}

        # Prepare form (pre-tanh TPR) and mask epsilons
        Y = distrib2local(X).narrow(1, 0, nfeature)
        Y = rearrange(Y, 'b f r -> b () f r')  # [nbatch x 1 x nftr x nrole]
        mask = epsilon_mask(X)
        mask = rearrange(mask, 'b r -> b () r')  # [nbatch x 1 x nrole]

        # Map unidimensional pattern weights to weights
        # on positive and negative feature values
        # (zero weight => underspecification)
        W = self.W(context) \
                .view(nbatch, npattern, nfeature)
        W = rearrange(W, 'b p f -> b p f ()')  # [nbatch x npattern x nftr x 1]
        #vals = torch.tensor([+3.0, -3.0, 0.0])  # approx. atanh [+1, -1, 0]
        #W = softmax(-(W - vals)**2.0, dim=-1)
        #Wpos, Wneg, Wzero = W.chunk(3, dim=-1)
        Wpos = softplus(W - 2.0, beta=2.0)  # xxx -1, 5
        Wneg = softplus(-W - 2.0, beta=2.0)  # xxx -1, 5
        #Wpos = relu(W - 1.0)
        #Wneg = relu(-W - 1.0)
        #print(f'Wpos {Wpos[0][0][:2].data.numpy()} {Wpos.shape}')
        #print(f'Wneg {Wneg[0][0][:2].data.numpy()}')
        #print(f'Wzero {Wzero[0][0][:2].data.numpy()}')

        # Compare symbol features to weighted feature patterns
        mismatch = Wpos * (1.0 - exp(match_pos(Y))) + \
                   Wneg * (1.0 - exp(match_neg(Y)))
        #print(f'match_pos {match_pos(Y).shape}')
        #sys.exit(0)
        mismatch = torch.sum(mismatch, 2)  # [nbatch x npattern x nrole]
        #mismatch = torch.mean(mismatch, 2)  # [nbatch x npattern x nrole]
        match = -mismatch + mask

        for (key, val) in [('W', W), ('Wpos', Wpos), ('Wneg', Wneg),
                           ('match', match)]:
            self.trace[key] = val.clone().detach()

        try:
            match_ = exp(match)
            assert np.all((0.0 <= match_.cpu().data.numpy())
                          & (match_.cpu().data.numpy() <= 1.0))
        except AssertionError as e:
            print(match_.cpu().data.numpy())
            print(Wpos[0, :, :, 0])
            print(Wneg[0, :, :, 0])
            print(X)
            raise

        return match

    def print(self, context):
        nbatch = context.shape[0]
        npattern = self.npattern
        nfeature = self.nfeature

        W = self.W(context) \
                .view(nbatch, npattern, nfeature)
        W = rearrange(W,
                      'b p f -> b p f ()')  # [nbatch x npattern x nfeature x 1]
        vals = torch.tensor([+3.0, -3.0, 0.0])  # {+1, -1, 0}
        W = softmax(-(W - vals)**2.0, dim=-1)
        Wpos, Wneg, Wzero = W.chunk(3, dim=-1)

        return {
            'Wpos': Wpos[0].detach().clone(),
            'Wneg': Wneg[0].detach().clone()
        }


# Basic feature-matching functions
def match_pos(X):
    """
    Gradient match to feature value +1 at each position in form X, computed elementwise in pre-tanh domain as sigmoid(5.0 * (x - 1.0)); 
    this gives approximately 0.5 match at x ≈ 1 (i.e., at tanh(1) ≈ .75). 
    Returns log match, apply exp() to output for values in (0,1)
    """
    match = logsigmoid(5.0 * (X - 1.0))
    return match


def match_neg(X):
    """
    Gradient match to feature value -1 at each position in form X, computed elementwise in pre-tanh domain as sigmoid(-5.0 * (x + 1.0)); this gives approximately 0.5 match at x ≈ -1 (i.e., at tanh(-1) ≈ -.75).
    Returns log match, apply exp() to output for values in (0,1)
    """
    match = logsigmoid(-5.0 * (X + 1.0))
    return match


# # # # # Deprecated # # # # #


class EndMatcher3(nn.Module):
    """
    Match at word edge ($)
    xxx assumes localist roles
    """

    def __init__(self, dcontext, nfeature=3, npattern=1):
        super(EndMatcher3, self).__init__()
        for posn in ['prev3', 'prev2', 'prev1']:
            setattr(self, f'matcher_{posn}',
                    LiteralMatcher(dcontext, nfeature, npattern))
        self.npattern = npattern

    def forward(self, X, context):
        nbatch = X.shape[0]
        #log_match_prev3 = shift(self.matcher_prev3(X, context), 3)
        #log_match_prev2 = shift(self.matcher_prev2(X, context), 2)
        log_match_prev1 = self.matcher_prev1(X, context)
        log_match_prev1 = shift(log_match_prev1, 1)

        match = torch.exp(log_match_prev1)  # [nbatch × npattern × nrole]
        mask = hardtanh0(X[:, 0])  # Epsilon mask
        stem_end = hardtanh(mask * -X[:, 1])  # Final delim mask
        match = match * stem_end.unsqueeze(
            1)  # Broadcast stem_end over patterns

        match = propagate(match, direction='LR->', mask=mask)[:, :, -1]
        #print(match.shape)
        return match


# xxx deprecate?
class Matcher(nn.Module):
    """
    Soft regex match of a single filler
    """

    def __init__(self, dcontext, nfeature):
        super(Matcher, self).__init__()
        self.context2w = nn.Linear(dcontext, nfeature, bias=True)
        self.context2b = nn.Linear(dcontext, 1, bias=True)
        self.context2tau = nn.Linear(dcontext, 1, bias=True)
        self.nfeature = nfeature

    def forward(self, X, context):
        w = self.context2w(context).unsqueeze(1)
        b = self.context2b(context).unsqueeze(1)
        tau = self.context2tau(context).unsqueeze(1)
        k = self.nfeature
        X = distrib2local(X)
        # log_match_i = tau * dot(w,x_i)
        score = torch.bmm(w, X.narrow(1, 0, k)) + b
        log_match = logsigmoid(tau * score).squeeze(1)

        # todo: Mask out matches to epsilon, in log domain
        return log_match


# todo: make subclass of matcher
class GCMMatcher(nn.Module):
    """
    Attention-weighted euclidean distance matcher,
    see GCM (Nosofsky 1986), ALCOVE (Kruschke 1991), etc. after Shepard (1962, 1987).
    If npattern==1, returns a single match with shape (nbatch, nrole)
    else (npattern>1) returns a bank of matches with shape (nbatch, nrole, npattern)
    """

    def __init__(self, dcontext, nfeature, npattern=1):
        super(GCMMatcher, self).__init__()
        self.context2W = \
            nn.Linear(dcontext, nfeature*npattern, bias=True)
        self.context2A = \
            nn.Linear(dcontext, nfeature*npattern, bias=True)
        self.context2tau = \
            nn.Linear(dcontext, npattern, bias=True)
        self.nfeature = nfeature
        self.npattern = npattern

    def forward(self, X, context):
        nbatch = X.shape[0]
        nfeature, npattern = self.nfeature, self.npattern
        # Feature specifications in [-1,+1] xxx check config.privative_ftrs
        W = tanh(self.context2W(context)) \
            .view(nbatch, nfeature, npattern)

        # Attention weights in [0,1]
        A = sigmoid(self.context2A(context)) \
            .view(nbatch, nfeature, npattern)

        # Sensitivity > 0
        tau = torch.exp(self.context2tau(context)).unsqueeze(1)
        k = self.nfeature
        if config.discretize:
            W = torch.round(W)
            A = torch.round(A)

        X = distrib2local(X)

        # Match of each filler determined by
        # attention-weighted euclidean distance
        score = pairwise_distance(X.narrow(1, 0, k), W, A)
        log_match = -tau * score

        # Mask out matches to epsilon in log domain
        # xxx move to matcher superclass
        log_match = log_match.squeeze(-1) + epsilon_mask(X)

        self.trace = {'score': score, 'log_match': log_match}
        return log_match

    # xxx move to superclass
    def compile(self, ftrspec, tau=1.0):
        """
        Compile instance from feature specifications 
        (see ftrspec2vec) and specificity
        """
        #w, a = ftrspec2vec(ftrspec)
        w, a = None, None  # xxx broken
        w = torch.FloatTensor(w)
        a = torch.FloatTensor(a)
        # note: weight specs correct for context2X nonlinearities
        self.context2W.weight.data[:, 0] = 10.0 * w
        self.context2A.weight.data[:, 0] = 20.0 * (a - 0.5)
        self.context2tau.weight.data[:] = tau
        self.context2W.bias.data[:] = 0.0
        self.context2A.bias.data[:] = 0.0
        self.context2tau.bias.data[:] = 0.0


# xxx deprecate
class LiteralMatcher1(nn.Module):
    """
    Weighted squared distance matcher inspired by classic models 
    for learning conjunctions of boolean literals, such as Literal.
    Assumes that each feature is binary (could be modified to accept 
    hard-coded set of values for each feature) with possibility of underspecification.
    If npattern == 1, returns a single match with shape (nbatch, nrole)
    else (npattern > 1) returns a bank of matches with shape (nbatch, nrole, npattern)
    # xxx npattern > 1 broken
    """

    def __init__(self, dcontext, nfeature, npattern=1, normalize=False):
        super(LiteralMatcher1, self).__init__()
        self.context2Wpos = \
            nn.Linear(dcontext, nfeature * npattern, bias = True)
        self.context2Wneg = \
            nn.Linear(dcontext, nfeature * npattern, bias = True)
        self.context2tau = \
            nn.Linear(dcontext, npattern, bias = True) # xxx not used
        self.nfeature = nfeature
        self.npattern = npattern
        self.normalize = normalize
        self.node = '-no-node-'

    def forward(self, X, context):
        nbatch = X.shape[0]
        nfeature, npattern = self.nfeature, self.npattern

        # Weights on positive (+1) and negative (-1) specifications
        # xxx clamp values to zero according to config.privative_ftrs
        # note: sigmoid nonlinearity for Wpos, Wneg because this is symmetric around 0,
        # exp nonlinearity for tau because can be unboundedly positive (could also be ReLU)
        # alternative: use ELU nonlinearity (or variants) for Wpos and Wneg, eliminate tau
        Wpos = softplus(self.context2Wpos(context) - 2.0) \
                .view(nbatch, nfeature, npattern)
        Wneg = softplus(self.context2Wneg(context) - 2.0) \
                .view(nbatch, nfeature, npattern)
        tau = torch.exp(self.context2tau(context))  # xxx not used

        # Normalize weights (with weight of underspecification fixed at unity)
        # xxx do this?
        if self.normalize:
            Z = Wpos + Wneg + 1.0
            Wpos = Wpos / Z
            Wneg = Wneg / Z

        Y = distrib2local(X)

        # Score (= log match) for each filler determined by
        # squared distance to ideal +/-1 values
        dist = Wpos * (Y.narrow(1,0,nfeature) - 1.0)**2.0 + \
               Wneg * (Y.narrow(1,0,nfeature) + 1.0)**2.0
        dist = torch.sum(dist, 1)  # xxx apply epsilon_mask here
        #log_match = -tau * dist
        log_match = -dist

        # Mask out matches to epsilon, in log domain
        # xxx move to superclass
        log_match = log_match + epsilon_mask(Y)

        self.trace = {'dist': dist, 'log_match': log_match}
        # xxx return match = exp(log_match) ?
        return log_match


# alternative similarity functions
# def forward(self, X, context):
#     w   = self.context2w(context)
#     b   = self.context2b(context)
#     tau = torch.exp(self.context2tau(context))
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
