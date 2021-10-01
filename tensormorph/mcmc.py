# -*- coding: utf-8 -*-

import copy, random, sys
from collections import namedtuple
#from pyhmc import hmc
#sys.path.append('/home/cwilso23/Library/Python/NUTS')
#import nuts
#import hamiltorch
#from Levenshtein import distance
import config
from tpr import *
import distributions as distrib
import data_util
from recorder import report
from torch.utils.data import DataLoader

StochasticParameter = \
    namedtuple('StochasticParameter', ['param', 'prior', 'prop'])


class MCMCSampler():

    def __init__(self):
        self.dataloader = DataLoader(
            config.dat_train,
            batch_size=2,  #config.batch_size,
            shuffle=True,
            collate_fn=data_util.morph_batcher)

    def sample(self):
        #stochparam_test = StochasticParameter(
        #    torch.tensor([0.0, 1.0]),
        #    distrib.discrete(torch.tensor([0.999, 0.001])),
        #    distrib.discrete(n=2))
        #param_test = stochparam_test.param
        #prior_test = stochparam_test.prior
        #print(prior_test.log_prob(param_test))
        #sys.exit(0)

        # Minibatch
        batch = next(iter(self.dataloader))
        print(batch)

        # Initialize grammar
        grammar = config.grammar
        #cogrammar = grammar.cogrammar
        #affixer = cogrammar.affixer
        #pivoter = affixer.pivoter
        # todo: stem truncator
        grammar.morph_attender.tau.data[:] = 2.0
        grammar.posn_attender.tau.data[:] = 2.0
        grammar.decoder.tau.data[:] = 2.0
        config.temperature = 1.0
        # todo: global sampling params
        optimizer = optim.Adam(grammar.parameters())
        #optimizer = None

        # Gather stochastic parameters
        stochastic = []
        stochastic_sizes = []
        search_size = 1
        for module in grammar.modules():
            if hasattr(module, 'stochastic'):
                module.init()
                #print(module, len(module.stochastic))
                stochastic += (module.stochastic)
                stochastic_sizes += [
                    x.param.shape[0] for x in module.stochastic
                ]
        #stochastic = [stochastic[1]]
        #stochastic_sizes = [stochastic_sizes[1]]
        print(f'{len(stochastic)} stochastic parameters')
        print(f'sizes: {stochastic_sizes}')
        for stochparam in stochastic:
            param = stochparam.param
            n = param.shape[0]
            n = n if n <= 5 else 5
            search_size *= n
            #print(param.data.numpy(), n)
        print(f'{search_size} possible cogrammars\n')
        #print(stochastic)
        self.grammar = grammar
        self.optimizer = optimizer
        self.batch = batch
        self.stochastic = stochastic
        self.stochastic_sizes = stochastic_sizes

        self.gradient_descent()
        energy = self.energy()
        print(energy)

        if 0:  # Hamiltonian Monte Carlo
            stochastic_size_all = np.sum(stochastic_sizes)
            x0 = torch.randn(stochastic_size_all)
            x0 = np.array(x0.data.numpy(), dtype=np.double)
            energy_func = self.energy_()
            #samples = hamiltorch.sample(log_prob_func=energy_func, params_init=x0,  num_samples=10)
            samples = hmc(energy_func,
                          x0=x0,
                          n_samples=200,
                          epsilon=0.5,
                          n_steps=5,
                          display=0,
                          return_diagnostics=1)
            #samples = nuts.nuts6(energy_func, 100, 100, x0, 0.5)
            #print(samples)

            batch['pred'] = grammar(batch)
            energy = self.energy()
            report(grammar, batch)
            print(energy)
            sys.exit(0)

        # Initialize chain
        energy = self.energy()
        print(energy)
        if 0:  # inspect initial gradients
            optimizer.zero_grad()
            energy.backward()
            for stochparam in stochastic:
                print(stochparam.param)
                print(stochparam.param.grad)
            sys.exit(0)

        # Sampler
        #move = torch.distributions.categorical.Categorical(
        #        logits=torch.ones(len(stochastic)))
        self.temperature = 1.0
        accept = torch.distributions.uniform.Uniform(0.0, 1.0)
        acceptance_rate = 0.0
        for i in range(config.max_samples):
            #random.shuffle(stochastic)
            if 0:  # Gibbs
                for k in range(len(stochastic)):
                    #    # xxx skip parameters with zero gradients?
                    #    self.propose1(stochastic, k)
                    self.propose_marginal(k)
                energy_new = self.energy()
                if energy.item() != energy_new.item():
                    print(energy.item(), '->', energy_new.item())
                energy = energy_new
            else:  # Metropolis-Hastings
                self.propose()
                energy_new = self.energy()
                a = self.accept_prob(energy_new, energy)
                if accept.sample() < a:
                    if energy.item() != energy_new.item():
                        print(energy.item(), '->', energy_new.item())
                    energy = energy_new
                    acceptance_rate += 1.0
                else:
                    self.revert(stochastic)

        print(f'acceptance rate {acceptance_rate/config.max_epochs}')
        batch['pred'] = grammar(batch)
        energy = self.energy()
        report(grammar, batch)
        print(energy)

    def energy(self):
        """
        Negative log posterior (aka 'energy')
        """
        batch, grammar, stochastic =\
            self.batch, self.grammar, self.stochastic
        #self.optimizer.zero_grad()
        self.grammar.zero_grad()
        self.batch['pred'] = grammar(batch)
        neglog_lik, neglog_liks = grammar.loglik_loss(batch['pred'],
                                                      batch['output'])
        #neglog_lik = neglog_lik / len(neglog_liks) # scale by batch size
        neglog_prior = 0.0
        for k in range(len(stochastic)):
            param = stochastic[k].param
            prior = stochastic[k].prior
            neglog_prior -= prior.log_prob(param)
        #print(neglog_lik, neglog_prior)
        energy = 100.0 * neglog_prior
        #energy = (neglog_lik + 0.1*neglog_prior)
        if 0:  # Levenshtein loss function
            self.batch['pred'].form_str = config.decoder.decode2string(
                self.batch['pred'].form)  # xxx use form_embedder instead
            energy = self.levenshtein_loss(self.batch['pred'].form_str[0],
                                           self.batch['output'].form_str[0],
                                           self.batch['output'].length[0])
            energy = torch.tensor([energy])
        return energy

    def energy_(self):

        def energy_func(x):
            # Unpack parameters
            i = 0
            for stochparam in self.stochastic:
                param = stochparam.param
                length = param.nelement()
                param.data = torch.FloatTensor(x[i:i + length])
                i += length
            # Calculate energy and gradient
            energy = self.energy()
            energy = -energy
            energy.backward()
            print(energy)
            grads = []
            for k, stochparam in enumerate(self.stochastic):
                grads.append(stochparam.param.grad)
            grad = torch.cat(grads, -1)
            grad = hardtanh(grad, -5.0, .0)
            #print(grad)
            energy = np.array(energy.data.numpy(), dtype=np.double)
            grad = np.array(grad.data.numpy(), dtype=np.double)
            print(x)
            print(grad)
            print(energy)
            return energy, grad

        return energy_func

    def propose(self):
        """
        Propose new values for all stochastic parameters
        """
        stochastic = self.stochastic
        self.prev_state = {}
        for k in range(len(stochastic)):
            param = stochastic[k].param
            prop = stochastic[k].prop
            self.prev_state[k] = param.data.clone()
            #proposal = prop.sample(param)
            n = param.shape[0]
            proposal = param + Normal(0.0, 0.5).sample(param.shape)
            proposal = hardtanh(proposal, -2.0, 2.0)  # BSB, baby
            param.data = proposal.data

    def propose1(self, k):
        """
        Propose new value for one stochastic parameter
        """
        #k = int(move.sample().item())
        stochastic = self.stochastic
        param = stochastic[k].param
        prop = stochastic[k].prop
        self.prev_state = {k: param.data.clone()}
        proposal = prop.sample(param)
        proposal.requires_grad = True
        param.data = proposal  #.data

    def propose_marginal(self, k):
        """
        Propose new value from marginal distribution on 
        one stochastic parameter
        """
        param = self.stochastic[k].param
        self.prev_state = {k: param.data.clone()}

        n = param.shape[0]
        n = n if n <= 5 else 5
        temperature = self.temperature
        marginal = torch.zeros(n)
        for i in range(n):
            param.data[:] = 0.0
            param.data[i] = 1.0
            marginal.data[i] = -self.energy()
        j = Categorical(logits=marginal / temperature).sample().item()
        #print(marginal.data, '->', j)
        param.data[:] = 0.0
        param.data[j] = 1.0

    def revert(self, stochastic):
        """
        Replace proposed value with previous value 
        for sampled stochastic parameter(s)
        """
        for k, val in self.prev_state.items():
            stochastic[k].param.data = val.data

    def accept_prob(self, energy_new, energy_old):
        """
        Metropolis-Hastings acceptance probability
        todo: add contribution from prior
        """
        temperature = self.temperature
        a = exp(-(energy_new - energy_old) / temperature)
        a = torch.min(torch.Tensor([a.item(), 1.0]))
        return a

    def gradient_descent(self, nsteps=40, epsilon=1.0e-1):
        """
        Initialize parameters with gradient descent 
        prior to MCMC sampling
        """
        optimizer = optim.Adagrad(self.grammar.parameters(), epsilon)
        for step in range(nsteps):
            energy = self.energy()
            energy.backward()
            optimizer.step()

    def levenshtein_loss(self, pred, targ, targ_len):
        pred = pred[:targ_len]
        targ = targ[:targ_len]
        dist = distance(pred, targ)
        print(pred, '|', targ, '|', dist)
        return dist