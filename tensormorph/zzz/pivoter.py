# -*- coding: utf-8 -*-

import config
from tpr import *
from radial_basis import GaussianPool
from matcher import WinnowMatcher
from inhibition import DirectionalInhibition


class BiPivoter(nn.Module):
    """
    Mark pivot locations (bidirectional)
    """
    def __init__(self, dcontext=1, nfeature=5):
        super(BiPivoter, self).__init__()
        # Select scan direction
        self.context2alpha = Linear(dcontext, 2)
        self.pivoter_LR = Pivoter(dcontext, nfeature, 'LR->')
        self.pivoter_RL = Pivoter(dcontext, nfeature, '<-RL')
        self.matcher = WinnowMatcher(1,1) # xxx not used


    def forward(self, form, context, **param):
        pivot_LR, pivots_LR = self.pivoter_LR(form, context)
        pivot_RL, pivots_RL = self.pivoter_RL(form, context)
        # Select LR-> or <-RL scan direction
        alpha = self.context2alpha(context)
        if config.sample:
            alpha = gumbel_softmax(
                logits = alpha, tau = 1.0, hard = True, dim = -1)
        else:
            alpha = softmax(alpha, dim = -1)
        pivot = alpha[:,0].unsqueeze(-1) * pivot_LR \
                + alpha[:,1].unsqueeze(-1) * pivot_RL
        return (pivot, (pivots_LR, pivots_RL))


    def init(self, direction='LR->', ftr='sym', before=True, bias=10.0, clamp=True):
        for p in self.context2alpha.parameters():
            p.data.fill_(0.0)
            p.requires_grad = (not clamp)
        if direction=='LR->':
            self.context2alpha.bias.data.fill_(bias)
            self.pivoter_LR.init(ftr, before, bias, clamp)
            self.pivoter_RL.init('sym', False, bias, clamp)
        elif direction=='<-RL':
            self.context2alpha.bias.data.fill_(-bias)
            self.pivoter_RL.init(ftr, before, bias, clamp)
            self.pivoter_LR.init('sym', False, bias, clamp)


class Pivoter(nn.Module):
    """
    Mark pivot locations (unidirectional)
    todo: add recording
    """
    def __init__(self, dcontext=1, nfeature=5, direction='LR->'):
        super(Pivoter, self).__init__()
        self.dcontext = dcontext
        self.nfeature = nfeature
        self.direction = direction
        # Select feature to determine pivot (e.g., begin, end, C, V)
        self.context2alpha = Linear(dcontext, nfeature)
        # Select pivot at|before each feature
        self.context2beta = Linear(dcontext, nfeature*2)
        self.matcher = WinnowMatcher(1,1) # xxx not used
        self.context2tau = torch.nn.Linear(dcontext, 1)


    def forward(self, form, context):
        nbatch = form.shape[0]
        nfeature = self.nfeature
        direction = self.direction
        tau = torch.exp(self.context2tau(context))

        if direction=='LR->':
            start_indx, end_indx, step = 0, config.nrole, 1
        elif direction=='<-RL':
            start_indx, end_indx, step = config.nrole-1, -1, -1

        # Pivot indicator for each feature across positions
        pivot_at = torch.zeros(nbatch, nfeature, config.nrole)
        # Flag indicating whether match to each feature already found
        found = torch.zeros(nbatch, nfeature)
        # Mask out epsilon fillers xxx see tpr.epsilon_mask
        mask = hardtanh(form[:,0,:], 0.0, 1.0).unsqueeze(-1)
        for i in range(start_indx, end_indx, step):
            # Euclidean distance^2 from positive values at ith position
            dist = (form[:,0:nfeature,i] - 1.0)**2.0
            # Similarity match xxx hard-coded precision
            match = torch.exp(-(5.0+tau) * dist) 
            # Pivot iff match & not-already-found & not-epsilon
            pivot_at[:,:,i] = match * (1.0 - found) * mask[:,i]
            # GRU-like update of flag
            found = match * 1.0 + (1.0 - match) * found
            #print(flag[0], '\n')
        pivot_at[:,0,:] = 0.0 # xxx ignore non-epsilon feature (redundant with edges) xxx hack used to encode no pivot
        #print(pivot_at[0]); sys.exit(0)
        # xxx block backprop if pivot finder is non-differentiable
        #pivot_at = pivot_at.detach()
        #pivot_at[pivot_at<0.5] = 0.0
        #pivot_at[pivot_at>0.5] = 1.0

        # Pivot-before is pivot-at shifted one position to left
        pivot_before = torch.cat((pivot_at[:,:,1:],
                                 torch.zeros(nbatch,nfeature,1)), -1)
        #print(pivot_at.shape, pivot_before.shape)

        # Select at|before for each feature and then feature
        alpha = self.context2alpha(context)
        beta = self.context2beta(context).view(nbatch, nfeature, 2)
        if config.sample:
            alpha = gumbel_softmax(
                logits = alpha, tau = 1.0, hard = True, dim = -1)
            beta = gumbel_softmax(
                logits = beta, tau = 1.0, hard = True, dim = -1)
        else:
            alpha = softmax(alpha, dim = -1)
            beta = softmax(beta, dim = -1)
        #print(alpha[0], beta[0])
        #print(pivot_at.shape, alpha.shape, beta.shape)
        pivot = beta[:,:,0].unsqueeze(-1) * pivot_at \
                + beta[:,:,1].unsqueeze(-1) * pivot_before
        pivot = torch.sum(alpha.unsqueeze(-1) * pivot, 1)
        #print(pivot.shape); sys.exit(0)
        #print(pivot[0])
        return (pivot, (pivot_at, pivot_before))
    

    def init(self, ftr='sym', before=True, bias=10.0, clamp=True):
        for m in [self.context2alpha, self.context2beta]:
            for p in m.parameters():
                p.data.fill_(0.0)
                p.requires_grad = (not clamp)
        ftr_indx = config.ftrs.index(ftr)
        self.context2alpha.bias.data[ftr_indx].fill_(bias)
        self.context2beta.bias.data[ftr_indx].fill_(
            bias * (1.0 if not before else -1.0))


class BiLSTMPivoter(nn.Module):
    """
    Find pivot with generic bidirectional recurrent layer
    """
    def __init__(self, dcontext=1, nfeature=5, npattern=1):
        super(BiLSTMPivoter, self).__init__()
        nhidden = 50
        self.rnn = torch.nn.LSTM(
            input_size = nfeature,
            hidden_size = nhidden,
            num_layers = 1,
            batch_first = True,
            bidirectional = True)
        self.linear = torch.nn.Linear(2*nhidden, 1)
        self.nfeature = nfeature
        self.matcher = WinnowMatcher(dcontext, nfeature) # xxx not used

    def forward(self, X, Context):
        Y = torch.nn.utils.rnn.pack_padded_sequence(
            X.narrow(1, 0, self.nfeature).transpose(1,2),
            X.slen,
            batch_first = True,
            enforce_sorted = False) # need to pack for <-RL scan
        activ, h_n = self.rnn(Y)
        activ, slen = torch.nn.utils.rnn.pad_packed_sequence(
            activ,
            batch_first = True,
            padding_value = 0.0,
            total_length = config.nrole) # [nbatch, nrole, 2*nhidden]
        #activ.view(nbatch, config.nrole, 2, nhidden) # outputs for each direction
        log_match = self.linear(activ).squeeze(-1) # [nbatch, nrole]
        log_match = log_match + epsilon_mask(X)
        #pivot = torch.sigmoid(log_match)
        pivot = torch.softmax(log_match, 1)
        return pivot, log_match, None


class EnsemblePivoter(nn.Module):
    """
    Ensemble multiple pivoters
    """
    def __init__(self, pivoters):
        super(EnsemblePivoter, self).__init__()
        n = len(pivoters)
        self.alpha = torch.nn.Parameter(torch.randn(n))
        self.pivoters = pivoters
        self.matcher = WinnowMatcher(1, 1) # xxx not used

    def forward(self, X, Context):
        beta = torch.softmax(self.alpha, 0)
        #alpha = torch.sigmoid(self.alpha)
        results = [pivoter(X,Context) for pivoter in self.pivoters]
        pivots = [pivot[0] for pivot in results]
        #for pivot in pivots:
        #    print(pivot[0][:5])
        #print([pivot.shape for pivot in pivots])
        pivots = torch.stack([pivot[0] for pivot in results], -1)
        #assert X.shape[0] == pivot.shape[0]
        #print(beta)
        #print(pivots.shape, beta.shape)
        pivot = pivots @ beta #torch.cat([alpha, 1.0 - alpha])
        #print(pivot[0])
        #sys.exit(0)
        return pivot, None, None


# # # # # older # # # # #

# xxx fixme
class Pivoter2(nn.Module):
    """
    Scanner specialized for restrictive pivot finding
    """
    def __init__(self, dcontext=1, nfeature=5, npattern=1):
        super(Pivoter2, self).__init__()
        # Single feature matrix specifying pivot
        self.matcher = WinnowMatcher(
            dcontext, nfeature, npattern)
        # Directional inhibition to scan LR-> or <-RL
        self.inhibiter = DirectionalInhibition()
        self.context2tau = nn.Linear(
            dcontext, 1, bias=True)
        # Gate to select first match LR-> or <-RL
        self.context2alpha = nn.Linear(
            dcontext, 1, bias=True)
        # Gate to pivot at or before first match
        self.context2beta = nn.Linear(
            dcontext, 1, bias=True)
    
    def forward(self, X, Context):
        nbatch, m, n = X.shape
        #tau = torch.exp(self.context2tau(Context))
        tau = torch.ones(1,1)
        alpha = torch.sigmoid(self.context2alpha(Context))
        beta = torch.sigmoid(self.context2beta(Context))

        # Find all matches to pattern
        match = torch.exp(self.matcher(X, Context))
        #print(match[0])

        # Apply directional inhibition LR-> and <-RL
        scan_LR, scan_RL = self.inhibiter(match, tau)
        scan = alpha * scan_LR + (1.0 - alpha) * scan_RL

        # Pivot at or immediately before the first match
        zero = torch.zeros((nbatch, 1))
        pivot = beta * scan + \
            (1.0 - beta) * torch.cat((scan[:,1:], zero), dim=1)

        #print(pivot[0])
        #pivot = pivot + epsilon_mask(X)
        pivot = torch.softmax(1.0 * torch.log(pivot), 1)
        #print(pivot[0])

        if config.recorder is not None:
            config.recorder.set_values(self.node, {
                'tau': tau, 
                'alpha': alpha,
                'beta': beta,
                'match': match,
                'scan_LR': scan_LR,
                'scan_RL': scan_RL,
                'scan': scan,
                'pivot': pivot
            })

        return pivot, scan, match


# xxx fixme
class Pivoter3(nn.Module):
    def __init__(self, dcontext=1, nfeature=5, npattern=1):
        super(Pivoter3, self).__init__()
        # Single feature matrix specifying pivot
        self.matcher = WinnowMatcher(
            dcontext, nfeature, npattern)
        # Directional inhibition to scan LR-> or <-RL
        self.inhibiter = DirectionalInhibition()
        self.context2tau = nn.Linear(
            dcontext, 1, bias=True)
        # Gate to select first match LR-> or <-RL
        self.context2alpha = nn.Linear(
            dcontext, 1, bias=True)
        # Gate to pivot at or before first match
        self.context2beta = nn.Linear(
            dcontext, 1, bias=True)

    def forward(self, X, Context):
        nbatch = X.shape[0]
        tau = torch.nn.functional.softplus(self.context2tau(Context))
        alpha = torch.sigmoid(self.context2alpha(Context))
        beta = torch.sigmoid(self.context2beta(Context))

        log_match = self.matcher(X, Context)
        log_match = torch.nn.functional.log_softmax(tau * log_match, 1)
        #print(log_match[0])
        scan_LR, scan_RL = self.inhibiter(log_match, tau)
        scan = alpha * scan_LR + (1.0 - alpha) * scan_RL
        #print(scan[0])
        zero = torch.zeros((nbatch, 1))
        pivot = beta * scan + \
            (1.0 - beta) * torch.cat((scan[:,1:], zero), dim=1)
        log_pivot = torch.nn.functional.log_softmax(tau * torch.log(pivot))
        pivot = torch.exp(log_pivot)
        return pivot, scan, torch.exp(log_match)


# xxx fixme
class FixedPivoter(Pivoter):
    """
    Hand-wired pivoter with a single softmax parameter
    - works with hand-specified weights for pre-first-vowel
    """
    def __init__(self, dcontext=1, nfeature=5, npattern=1,
        fixed_state_dict=None):
        super(FixedPivoter, self).__init__()
        #self.pivoter = Pivoter(dcontext, nfeature, npattern)
        for param in self.parameters():
            param.requires_grad = False
            param.fill_(0.0)
        self.load_state_dict(fixed_state_dict, strict=False)
        self.tau = torch.ones(1) * 10.0 # todo: fit this
        self.matcher = WinnowMatcher(1,1) # xxx not used
        self.node = '-no-node-'

    def forward(self, X, Context):
        tau = self.tau
        pivot1, scan, match = super().forward(X, Context)
        pivot2 = torch.softmax(tau * torch.log(pivot1), 1)
        #print(tau)
        #print(pivot1[0]); print(scan[0]); print(match[0])
        #print(tau * torch.log(pivot1[0]))
        #print('pivot2:', pivot2[0])
        return pivot2, scan, match


class FixedPivoterHard(nn.Module):
    """
    Hand-wired pivoter with no parameters
    """
    def __init__(self, feature, value, direction='LR->', before=False):
        super(FixedPivoterHard, self).__init__()
        self.feature = feature
        self.value = value
        self.before = before
        self.matcher = WinnowMatcher(1,1) # xxx not used
    def forward(self, X, Context):
        feature = self.feature
        value = self.value
        before = self.before
        nbatch = X.shape[0]
        match = torch.zeros(nbatch)
        flag = torch.ones(nbatch)
        pivot = torch.zeros(nbatch, config.nrole)
        for i in range(X.shape[2]):
            match.fill_(0.0)
            match.masked_fill_(X[:,feature,i]==value, 1.0)
            pivot[:,i] = match * flag
            flag.masked_fill_(match==1.0, 0.0)
        if before:
            pivot = torch.cat([pivot[:,1:], torch.zeros(nbatch,1)],1)
        return pivot, None, None