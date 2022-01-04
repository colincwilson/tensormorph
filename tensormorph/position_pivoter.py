# -*- coding: utf-8 -*-
import config
from tpr import *
from scanner import *
from syllable_parser import SyllableParser
#from prosodic_parser import *
#import distributions as distrib
#import mcmc


class PositionPivoter(nn.Module):
    """
    Pivot function defined by edge positions 
    (Ultan/Moravcsik/Anderson/Yu)
    """

    def __init__(self):
        super(PositionPivoter, self).__init__()
        # dcontext = config.dcontext
        self.pivots = ['none', 'after ⋊', 'before ⋉'] \
                    + [f'{lcn} {drn} {elt}'
                        for lcn in ('before', 'after')
                        for drn in ('first', 'last')
                        for elt in ('C', 'V', 'X')] \
                    + ['after first σ', 'before last σ'] \
                    + ['circumfix1'] #, 'circumfix2']
        self.npivot = len(self.pivots)
        #self.context2w = Linear(dcontext, self.npivot) # pivot weights
        bias = (config.dcontext > 1)

        #self.context2w = nn.Sequential(
        #nn.Linear(config.dcontext, config.dcontext),
        #nn.Tanh(),
        #nn.Linear(config.dcontext, self.npivot, bias=bias))
        #    Multilinear(config.context_size, 30, self.npivot, bias=bias) )
        #if dcontext == 1: # remove redundant weight parameters
        #    self.context2w.weight.detach_()
        self.parser = SyllableParser()

    def forward(self, base, W=None):
        form = base.form
        nbatch = form.shape[0]

        # Parse form up to syllable level
        parse = self.parser(form)
        #print(parse.shape); sys.exit(0)
        after_first = inhibit(parse, 'LR->')
        after_last = inhibit(parse, '<-RL')
        before_first = shift1(after_first, -1)
        before_last = shift1(after_last, -1)

        # Assemble pivots (see syllable parser for element indices)
        pivots = [
            torch.zeros(
                nbatch,
                1,
                config.nrole,
                requires_grad=False,
                device=config.device),  # No pivot
            after_first[:, 0].unsqueeze(1),  # After first ⋊
            # before_last[:,1].unsqueeze(1),        # Before last ⋉
            before_first[:, 1].unsqueeze(1),  # Before first ⋉
            before_first[:, 2:5],  # Before first C | V | X
            before_last[:, 2:5],  # Before last C | V | X
            after_first[:, 2:5],  # After first C | V | X
            after_last[:, 2:5],  # After last C | V | X
            after_first[:, 6].unsqueeze(1),  # After first syll)
            before_last[:, 5].unsqueeze(1),  # Before last (syll
            hardtanh0(after_first[:, 0] + before_first[:, 1]).unsqueeze(1),
            # Circumfix1
            #hardtanh0(before_first[:,4] + after_last[:,4]).unsqueeze(1),
            #                                       # Circumfix2
        ]
        pivots = torch.cat(pivots, 1).detach()  # [nbatch x npivot x nrole]
        # xxx document detach()

        assert not np.any(np.isnan(pivots.cpu().data.numpy())), \
                f'pivot value is nan {pivots}'
        assert np.all((0.0 <= pivots.cpu().data.numpy()) &
                      (pivots.cpu().data.numpy() <= 1.0)), \
                f'pivot value outside [0,1] {pivots}'

        # Return pivots for subsequent selection
        # todo: make default
        if W is None:
            return pivots

        # Select pivot for each affix
        pivot = einsum('bpi,ap->bai', pivots, W)
        assert not np.any(np.isnan(pivot.cpu().data.numpy())), \
                f'pivot value is nan: pivot = {pivot}, W={W}'
        # [nbatch x naffix x nrole]
        return pivot

    # # # # # Deprecated # # # # #

    def forward1(self, base, context):
        #form = distrib2local(stem.form)
        #mask = hardtanh0(form[:,0,:])
        form = base.form
        nbatch = form.shape[0]

        # Parse form up to syllable level
        parse = self.parser(form)
        #print(parse.shape); sys.exit(0)
        first = inhibit(parse, 'LR->')
        last = inhibit(parse, '<-RL')
        before_first = shift1(first, -1)
        before_last = shift1(last, -1)

        # Assemble pivots (see syllable parser for element indices)
        pivots = [
            torch.zeros(
                nbatch,
                1,
                config.nrole,
                requires_grad=False,
                device=config.device),  # No pivot
            first[:, 0].unsqueeze(1),  # After first ⋊
            # before_last[:,1].unsqueeze(1),        # Before last ⋉
            before_first[:, 1].unsqueeze(1),  # Before first ⋉
            before_first[:, 2:5],  # Before first C | V | X
            before_last[:, 2:5],  # Before last C | V | X
            first[:, 2:5],  # After first C | V | X
            last[:, 2:5],  # After last C | V | X
            first[:, 6].unsqueeze(1),  # After first syll)
            before_last[:, 5].unsqueeze(1),  # Before last (syll
            hardtanh0(first[:, 0] +
                      before_first[:, 1]).unsqueeze(1),  # Circumfix1
            #hardtanh0(first[:,4] + before_last[:,4]).unsqueeze(1), # Circumfix2
        ]
        pivots = torch.cat(pivots, 1).transpose(2, 1)
        pivot = pivots.detach()  # xxx document

        # Select pivot
        #w = distrib.rsample(self.context2w(context)) # xxx incorrect softmax dim
        w = softmax(self.context2w(context), dim=1)  # xxx check dim
        #print(pivots.shape, w.shape); sys.exit(0)
        #pivot = mask * bmatvec(pivots, w)
        pivot = bmatvec(pivots, w)
        return pivot

    # xxx no longer used
    def init(self):
        self.stochastic = []
        self.context2alpha.weight.data.fill_(0.0)
        self.context2alpha.bias.data.fill_(0.0)
        self.context2alpha.bias.data[0] = 1.0
        # Register stochastic params
        self.stochastic = [
            mcmc.StochasticParameter(self.context2alpha.bias,
                                     distrib.SphericalNormal(n=self.npivot),
                                     distrib.Discrete(n=self.npivot)),
        ]
