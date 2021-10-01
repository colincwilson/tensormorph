# -*- coding: utf-8 -*-

import config
from tpr import *


class ProsodicParser(nn.Module):
    """
    Build a single maximal syllable and maximal foot at the 
    left or right edge of a form. Currently has no trainable 
    parameters, implementing typologically common prosodic structure
    Assumes:
    - Epsilon-free input sequence aligned at left edge, with or 
    without begin and end delimiter
    - Onsets obligatory except at initial edge
    - Singleton codas ok but optional
    - Complex onsets and codas prohibited except at edges
    """

    def __init__(self):
        super(ProsodicParser, self).__init__()
        # Threshold for privative features
        self.theta = torch.tensor(0.25)

    def forward(self, form, direction='LR->'):
        """
        Mark begin and end of maximal syllable and foot,
        scanning in the designated direction. Also mark 
        begin delimiter if present in initial position, 
        first consonant before any vowel if present, 
        and first vowel
        """
        nbatch = form.shape[0]
        nrole = config.nrole
        theta = self.theta

        # Threshold sym, begin/end, C/V features
        mask = form[:, 0, :].clone().detach()
        begin = form[:, 1, :].clone().detach()
        end = -form[:, 1, :].clone().detach()
        C = form[:, 2, :].clone().detach()
        V = -form[:, 2, :].clone().detach()
        for x in [mask, begin, end, C, V]:
            threshold_(x, theta, 0.0)

        # Initialize edges and other landmarks
        parse = {
            x: torch.zeros(nbatch, nrole) for x in [
                '(syll1', 'syll1)', '(syll2', 'syll2)', 'begin', 'end', 'C1',
                'V1'
            ]
        }
        syll_state = torch.zeros(nbatch)  # State of syllable parser
        c_state = torch.zeros(nbatch)  # State of C1 parser
        v_state = torch.zeros(nbatch)  # State of V1 parser
        end_state = torch.zeros(nbatch)  # State of end_delim parser
        V_count = torch.sum(V * mask, 1)  # Vowel count lookahead

        # Mark begin delimiter if present in initial position
        parse['begin'][:, 0] = begin[:, 0] * mask[:, 0]

        # Mark syllable edges and C1, V1
        if direction == 'LR->':
            idx_start, idx_end, idx_step = 0, nrole, 1
        else:
            idx_start, idx_end, idx_step = nrole - 1, -1, -1

        for i in range(idx_start, idx_end, idx_step):
            # Is next symbol a consonant? a vowel? at end?
            j = (i + idx_step)
            C_next = torch.zeros(nbatch) if (j == idx_end) \
                else C[:,j] * mask[:,j]
            V_next = torch.zeros(nbatch) if (j == idx_end) \
                else V[:,j] * mask[:,j]
            X_next = hardtanh(C_next + V_next, 0.0, 1.0)

            # Is current symbol a consonant? a vowel? at end?
            C_match = C[:, i] * mask[:, i]
            V_match = V[:, i] * mask[:, i]
            X_match = hardtanh(C_match + V_match, 0.0, 1.0)
            X_end = X_match if (j == idx_end) \
                else 1.0 - X_next

            # # # # # Syllable parser # # # # #
            # Beginning of first syllable: first C or V
            syll1_begin = (syll_state == 0) * X_match
            syll_state += syll1_begin  # q0 -> q1

            # Nucleus of first syllable
            syll_state += (syll_state == 1) * V_match  # q1 -> q2

            # End of first syllable
            syll1_end = (syll_state == 2) * hardtanh(
                (
                    # Nucleus if no following C
                    (V_count > 1) * V_match * (1.0 - C_next)
                    # First C after nucleus
                    + (V_count > 1) * C_match
                    # Last segment of monosyllable
                    + (V_count == 1) * X_end),
                0.0,
                1.0)
            syll_state += syll1_end  # q2 -> q3

            # Beginning of second syllable: first C or V after syll1
            syll2_begin = (syll_state == 3) * (1.0 - syll1_end) * X_match
            syll_state += syll2_begin  # q3 -> q4

            # Nucleus of second syllable
            syll_state += (syll_state == 4) * V_match  # q4 -> q5

            # End of second syllable
            syll2_end = (syll_state == 5) * hardtanh(
                (
                    # Nucleus if no following C
                    (V_count > 2) * V_match * (1.0 - C_next)
                    # First C after nucleus
                    + (V_count > 2) * C_match
                    # Last segment of bisyllable
                    + (V_count == 2) * X_end),
                0.0,
                1.0)
            syll_state += syll2_end  # q5 -> q6

            parse['(syll1'][:, i] = syll1_begin
            parse['syll1)'][:, i] = syll1_end
            parse['(syll2'][:, i] = syll2_begin
            parse['syll2)'][:, i] = syll2_end

            # # # # # C1/V1/end parser # # # # #
            # First C if not preceded by V
            C1 = (c_state == 0) * (v_state == 0) * C_match
            c_state += C1  # q0 -> q1

            # First V
            V1 = (v_state == 0) * V_match
            v_state += V1  # q0 -> q1

            # First instance of end delimiter
            end_match = (end_state == 0) * end[:, i] * mask[:, i]
            end_state += end_match  # q0 -> q1

            parse['C1'][:, i] = C1
            parse['V1'][:, i] = V1
            parse['end'][:, i] = end_match

        # Align foot boundaries with syllable boundaries
        parse['(foot1'] = parse['(syll1']
        parse['foot1)'] = (V_count.unsqueeze(-1) > 1) * parse['syll2)'] \
                        + (V_count.unsqueeze(-1) < 2) * parse['syll1)']

        # Convention: if C1 is not present (equiv., terminal c_state is 0),
        # mark begin delim (LR->) or end delim (<-RL)
        c_state = c_state.unsqueeze(-1)
        if direction == 'LR->':
            parse['C1'] = (c_state == 1) * parse['C1'] \
                          + (c_state == 0) * parse['begin']
        else:
            parse['C1'] = (c_state == 1) * parse['C1'] \
                          + (c_state == 0) * parse['end']

        # Reverse brackets for right-to-left scanning
        if direction == '<-RL':
            for (l, r) in [('(syll1', 'syll1)'), ('(syll2', 'syll2)'),
                           ('(foot1', 'foot1)')]:
                parse[l], parse[r] = parse[r], parse[l]

        return parse

    def show(self, morph, direction='LR->'):
        """
        Mark syllable and foot boundaries, C1, V1
        """
        parse = self(morph.form, direction)
        parse = {key: val[0] for key, val in parse.items()}
        form_mark = morph.form_str[0].split()
        for i in range(len(form_mark)):
            seg = form_mark[i]
            if parse['C1'][i] or parse['V1'][i]:
                seg = seg + ' •'
            if parse['(syll1'][i] or parse['(syll2'][i]:
                seg = '[' + seg
            if parse['syll1)'][i] or parse['syll2)'][i]:
                seg = seg + ']'
            if parse['(foot1'][i]:
                seg = '(' + seg
            if parse['foot1)'][i]:
                seg = seg + ')'
            form_mark[i] = seg
        return ' '.join(form_mark)
