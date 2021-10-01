# -*- coding: utf-8 -*-

from tpr import *
import cogrammar

class EnsembleCogrammar(nn.Module):
    """
    Ensemble of cogrammars with softmax pooling
    xxx non-reduplication only
    xxx fix recorder interface
    """
    def __init__(self, nensemble):
        super(EnsembleCogrammar, self).__init__()
        self.nensemble = nensemble
        self.alpha = torch.nn.Parameter(torch.randn(nensemble))
        self.cogrammar = [cogrammar.Cogrammar() 
            for i in range(nensemble)]

    def forward(self, Stem, Morphosyn, max_len):
        Outputs = [self.cogrammar[i](Stem, Morphosyn, max_len)
                    for i in range(self.nensemble)]
        alpha = torch.softmax(self.alpha, 0)
        Output = torch.zeros_like(Outputs[0])
        for i in range(self.nensemble):
            Output += alpha[i] * Outputs[i]
        return Output