# -*- coding: utf-8 -*-

# Containers for morphs, and morph combination + truncation
# with hierarchical attention
# note: structured alternative to legacy Combiner, Writer

import config
from tpr import *


class MorphOp(nn.Module):
    """
    Morph combination and truncation with hierarchical attention
    note: no trainable parameters
    todo: possibly make read()/write() member funcs of Morph, for scaling 
    to combinations of > 2 morphs (e.g., softmax over affix vocab)
    """

    def __init__(self):
        super(MorphOp, self).__init__()
        self.morph_indx = None  # soft index to stem (0) or affix (1)
        self.base_posn = None  # soft scalar position within base,
        self.affix_posn = None  # affix,
        self.output_posn = None  # output
        self.trace = None

    def forward(self, base, affix):
        """
        Map stem and affix morphs to output morph
        """
        output = self.reset(base, affix)
        morph_indx, base_posn, affix_posn, output_posn = \
            self.morph_indx, self.base_posn, \
            self.affix_posn, self.output_posn

        for t in range(config.nrole + 6):  # xxx ndecode
            # Map scalar morph index to morph attention
            morph_attn = config.morph_attender(morph_indx)
            base_attn, affix_attn = morph_attn.chunk(chunks=2, dim=-1)
            morph_state = {'morph_indx': morph_indx, 'morph_attn': morph_attn}

            # Read filler, pivot, copy from stem, affix
            base_state = self.read(base, base_posn)
            affix_state = self.read(affix, affix_posn)

            # Convex combo of filler and copy from stem, affix
            fill = base_attn * base_state['filler'] \
                    + affix_attn * affix_state['filler']
            copy = base_attn * base_state['copy'] \
                    + affix_attn * affix_state['copy']

            # Write to output
            output_state = \
                self.write(output, output_posn, fill, copy)

            # Trace internal processing
            self.update_trace(morph_state, base_state, affix_state,
                              output_state)

            # Update scalar morph index, switching at pivots
            morph_indx = morph_indx \
                + base_attn * base_state['pivot'] \
                - affix_attn * affix_state['pivot']

            # Update scalar positions within morphs,
            # advancing in fractions of unit steps
            base_posn = base_posn + base_attn * 1.0
            affix_posn = affix_posn + affix_attn * 1.0
            output_posn = output_posn + copy * 1.0

        # Normalize output fillers by total attention to roles
        # xxx localist roles only
        #output.form = output.form / self.attn_total.unsqueeze(1)

        return output

    def read(self, morph, posn):
        """
        Read (unbind) attributes of morph at soft position
        """
        # Map scalar position to attention over positions
        posn_attn = config.posn_attender(posn)

        # Read (unbind) current filler, pivot, copy
        f = attn_unbind(morph.form, posn_attn)
        p = attn_unbind(morph.pivot, posn_attn)
        c = attn_unbind(morph.copy, posn_attn)

        state = {
            'posn': posn,
            'posn_attn': posn_attn,
            'filler': f,
            'pivot': p,
            'copy': c
        }
        return state

    def write(self, morph, posn, f, c):
        """
        Write (bind) filler f or epsilon to  
        soft position in morph
        """
        # Map scalar position to attention
        posn_attn = config.posn_attender(posn)
        self.attn_total = self.attn_total + posn_attn

        # Accumulate f/r binding -or- epsilon
        f = c * f
        morph.form, r = \
            attn_bind(morph.form, f, posn_attn)

        state = {
            'posn': posn,
            'posn_attn': posn,
            'fill': f,
            'role': r,
            'copy': c
        }

        return state

    def reset(self, base, affix):
        """
        Initialize/reset output, morph_indx, morph posns, trace 
        """
        nbatch = base.form.shape[0]
        output_form = torch.zeros((nbatch, config.dsym, config.nrole),
                                  requires_grad=True,
                                  device=config.device)
        output = Morph(output_form)
        for x in ['morph_indx', 'base_posn', 'affix_posn', 'output_posn']:
            setattr(
                self, x,
                torch.zeros((nbatch, 1),
                            requires_grad=True,
                            device=config.device))
        # Track total attention to output roles over write steps
        self.attn_total = 1.0e-7 * torch.ones(
            (nbatch, config.nrole), requires_grad=True, device=config.device)
        # xxx experimental
        #setattr(self, 'affix_posn', affix.begin)
        return output

    def update_trace(self, morph_state, base_state, affix_state, output_state):
        """
        Trace internal processing for debugging / visualization
        """
        if self.trace is None:
            self.trace = {}
        trace = self.trace
        for (prefix,
             d) in zip(['', 'base_', 'affix_', 'output_'],
                       [morph_state, base_state, affix_state, output_state]):
            for key, val in d.items():
                key = prefix + key
                if key not in trace:
                    trace[key] = []
                trace[key].append(val.clone().detach())


class Morph(nn.Module):
    """
    Container for form embedding matrix, form symbol ids, 
    pivot and copy vectors, and other attributes of batch
    and other attributes of a batch of morphs
    (alternative to binding as composite tpr)
    note: no trainable parameters, null forward()
    """

    def __init__(self,
                 form=None,
                 form_id=None,
                 form_str=None,
                 length=None,
                 pivot=None,
                 copy=None):
        super(Morph, self).__init__()
        self.form = form  # symbol/position embeddings
        # xxx .to(device=config.device)
        self.form_id = form_id  # symbol ids
        self.form_str = form_str  # string
        self.length = length  # string length
        self.pivot = pivot  # pivot vector
        self.copy = copy  # copy vector

    def forward(self):
        return None

    def _str(self, markup=True, trim=True):
        """
        String representation of batch of morphs, 
        with available markup as available
        """
        if self.form is None:
            return self.form_str
        # Create string if it does not exist or have markup to add
        if (self.form_str is None) or (self.copy is not None) \
           or (self.pivot is not None):
            copy = self.copy if markup else None
            pivot = self.pivot if markup else None
            form_str = config.form_embedder.tpr2string(
                self.form, copy=copy, pivot=pivot, trim=trim)
            return form_str
        # Otherwise use existing string xxx todo: modulate by stem vs. output
        return self.form_str
