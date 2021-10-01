# -*- coding: utf-8 -*-

import config
from tpr import *
from distance import euclid_batch, sqeuclid_batch
#from torch_struct import AlignmentCRF


class Decoder(nn.Module):
    """
    Decode localist or distributed TPR to (unnormalized) 
    log probabilities of discrete symbols
    todo: add beam decoding
    """

    def __init__(self, tau=None, tau_min=None):
        super(Decoder, self).__init__()
        if tau is not None:
            self.tau = Parameter(0.1 * torch.randn(1) * tau)
        else:
            self.tau = Parameter(torch.rand(1) - 0.5)
            #self.tau = Parameter(-2.0 * torch.ones(1))
        if tau_min is not None:
            self.tau_min = tau_min
        else:
            self.tau_min = 0.0
        #self.tau = Parameter(-5.0 * torch.ones(1))
        #self.tau = Parameter(torch.zeros(1), requires_grad=False)  # xxx clamped precision
        self.eps = 1.0e-8
        self.add_noise = False

    def forward(self, Y):
        """
        Unnormalized log probabilities of discrete symbols at 
        each position in tanh(Y)
        """
        tau = exp(self.tau) + self.tau_min  # config.tau_min
        # xxx cap decoder precision to avoid dependence on very small differences in filler - symbol similarity
        tau = relu6(tau)
        #tau = 2.0 * torch.sigmoid(self.tau)
        # Transform to localist filler/role bindings
        # Y = Y @ config.U # xxx check with distributed roles
        Y = distrib2local(Y)  # [nbatch x dsym x nrole]
        # Add noise before tanh non-linearity, attempting to reduce reliance on very small differences in filler-symbol similarity
        if self.add_noise:
            # noise ~ Normal(0, 1)
            #Y = Y + torch.randn_like(Y)
            # noise ~ Uniform[-1/2, +1/2]
            Y = Y + (1.0 * (torch.rand_like(Y) - 0.5))
        # Map from real (pre-tanh) domain into (-1,+1)
        Y = tanh(Y)
        # Compare fillers to discrete symbol embeddings
        dist = sqeuclid_batch(Y, config.F)  # [nbatch x nsym x nrole]
        sim = -tau * (dist + self.eps)

        self.trace = {'tau': tau.clone().detach()}
        return sim


# # # # # Deprecated # # # # #


class LocalistAlignmentDecoder(nn.Module):
    """
    Alignment-based decoder implemented with pytorch-struct
    xxx assumes local role representations
    xxx not better (probably worse) than LocalistDecoder
    """

    def __init__(self, tau=None):
        super(LocalistAlignmentDecoder, self).__init__()
        self.tau = Parameter(torch.randn(1)*tau) if tau\
              else Parameter(torch.randn(1)) # xxx not used
        self.localist_decoder = LocalistDecoder(tau)

    def forward(self, Y, Z, lengths):
        """
        Y - batch of predictions (tprs to be decoded)
        Z - batch of target outputs (target indices)
        lengths - target output lengths for masking
        """
        nbatch = Y.shape[0]
        max_len = torch.max(lengths).item()  # max length in batch
        indel = -20.0  # xxx cost of insertion/deletion in log space

        # Log probability (normalized) of each possible symbol
        # in each role, tensor [nbatch x nrole x nsym]
        sim = -1.0 * sqeuclid_batch(Y, config.F).transpose(2, 1)
        logprob = sim  # xxx don't normalize
        #logprob = torch.nn.functional.log_softmax(sim, 1)
        #logprob = logprob[:,:max_len,:]
        #print(logprob.shape, Z.shape, logprob[Z.unsqueeze(-1)].shape)

        # Log potentials for emission and insertion/deletion
        log_potentials = torch.zeros((nbatch, max_len, max_len, 3),
                                     config.device)
        #log_potentials[:,:,:,0].fill_(indel) # xxx check
        #log_potentials[:,:,:,2].fill_(indel)
        for b in range(nbatch):  # xxx vectorize
            for i in range(max_len):
                for j in range(max_len):
                    log_potentials[b, i, j, :] = logprob[b, i, Z[b, j]]
        np.save('/Users/colin/Desktop/log_potentials.npy',
                log_potentials.data.numpy())
        #print(log_potentials[0]); sys.exit(0)
        #for b in range(nbatch): # todo: vectorizer over batch
        #    log_potentials[b,:,:,1] = logprob[b,:,Z[b,:max_len]]

        # Alignment
        alignment = AlignmentCRF(
            log_potentials, local=True, lengths=lengths, max_gap=1)
        # Log sum of all alignments for each batch member
        return alignment.partition

    def decode(self, form):
        return self.localist_decoder.decode(form)
