# -*- coding: utf-8 -*-

import config
import torch


class Prior():

    def __init__(self):
        self.affix_length_p = \
            0.01 * torch.arange(0, config.nrole,
                        dtype=torch.float, requires_grad=False)
        self.total_val = 0.0

    def prior_val(self):
        val = self.total_val
        self.total_val = 0.0
        return val

    def affix_length_prior(self, affix_pivot):
        val = affix_pivot @ self.affix_length_p
        self.total_val += torch.sum(val)

    def morphosyn_slot_prior(self, dim_attn):
        # GSC prior
        val = torch.sum(dim_attn * (1.0 - dim_attn)) + \
              torch.relu(torch.sum(dim_attn) - 1.0)
        self.total_val += 0.1 * val
