# -*- coding: utf-8 -*-

import config
from tpr import *
from matcher import match_pos, match_neg


class SyllableParser(nn.Module):
    """
    Encode syllable edges (begin, end) assuming typologically 
    common syllable structure:
    - Complex onsets and codas prohibited except at word edges (VCCV -> VC.CV)
    - Onset preference (VCV -> V.CV)
    - No complex nuclei (VV -> V.V)
    For convenience, also encodes begin / end / C / V / X locations
    todo: move computations to log-linear domain
    """

    def __init__(self):
        super(SyllableParser, self).__init__()

    def forward(self, form):
        nbatch = form.shape[0]
        form = distrib2local(form)
        form = form.detach()  # xxx document

        #theta = self.theta
        ftr = unbind_ftr(form, 0, 3)  # Sym, begin/end, C/V features
        if 1:  # Graded threshold
            mask = epsilon_mask(form)  # Epsilon mask
            begin = exp(mask + match_pos(ftr[:, 1]))  # Initial delim (⋊)
            end = exp(mask + match_neg(ftr[:, 1]))  # Final delim (⋉)
            c = exp(mask + match_pos(ftr[:, 2]))  # Consonant (C)
            v = exp(mask + match_neg(ftr[:, 2]))  # Vowel (V)
            seg = exp(mask + match_pos(torch.abs(ftr[:, 2])))  # Segment (X)
        else:  # Hard threshold (pre-tanh domain)
            theta = 1.0
            mask = hardtanh0((ftr[:, 0] > theta)).float()
            begin = mask * (ftr[:, 1] > theta).float()
            end = mask * (-ftr[:, 1] > theta).float()
            c = mask * (ftr[:, 2] > theta).float()
            v = mask * (-ftr[:, 2] > theta).float()
            seg = mask * (torch.abs(ftr[:, 2]) > theta).float()
        #for x in [begin, end, c, v]:
        #    print(np.all(x.data.numpy() < 1.0))

        #seg = c + v # Consonant or Vowel xxx check
        begin_prev = shift1(begin, k=1)  # Begin delim before current position
        end_next = shift1(end, k=-1)  # End delim after current position
        c_prev = shift1(c, k=1)  # Consonant before current position
        c_next = shift1(c, k=-1)  # Consonant after current position
        v_prev = shift1(v, k=1)  # Vowel before current position
        v_prev2 = shift1(v_prev, k=1)  # Vowel two before current position
        v_next = shift1(v, k=-1)  # Vowel after current position
        v_next2 = shift1(v_next, k=-1)  # Vowel two after current position

        # Positions at which syllables begin
        syll_begin = (
            v_prev * c * v_next  # C in V_V
            + v_prev2 * c_prev * c * v_next  # C in VC_V
            + v_prev * v  # V in V_
        )
        syll_begin = (1.0 - syll_begin) * begin_prev * seg  # First segment

        # Positions at which syllables end
        syll_end = (
            v * c_next * v_next2  # V in _CV
            + v_prev * c * c_next * v_next2  # C in V_CV
            + v * v_next  # V in _V
        )
        syll_end = (1.0 - syll_end) * seg * end_next  # Last segment

        # Gradient parse, values in (0, 1)
        parse = torch.stack([begin, end, c, v, seg, syll_begin, syll_end], 1)
        #print(np.all(parse.data.numpy() >= 0.0) and
        #      np.all(parse.data.numpy() <= 1.0))
        #print(np.round(parse.data[0].numpy(), 2)); sys.exit(0)

        assert not np.any(np.isnan(parse.cpu().data.numpy())), \
                f'parse value is nan {parse}'
        assert np.all((parse.data.numpy() >= 0.0) &
                      (np.all(parse.data.numpy() <= 1.0))), \
                f'parse value outside [0,1] {parse}'

        return parse
