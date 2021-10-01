# -*- coding: utf-8 -*-
# Directional scanning and associated operations
# (directional propagation, directional inhibition)

import config
from tpr import *
from tpr import bdot


def scan(X, direction='both', inclusive=True):
    """
    All-prefix-sum (LR->) and/or all-suffix-sum (<-RL), 
    inclusive (1, default) or exclusive (0), applied 
    along role dimension of X [nbatch x nftr/npattern x nrole]
    (batch operation; assumes localist roles)
    see https://en.wikipedia.org/wiki/Prefix_sum 
    see Blelloch 1990, etc. on parallel computation 
    and applications; used here for inhibition/propagation
    xxx add optional mask
    """
    # Apply bidirectionally
    if direction == 'both':
        scan_LR = scan(X, 'LR->', inclusive)
        scan_RL = scan(X, '<-RL', inclusive)
        return (scan_LR, scan_RL)

    # Apply unidirectionally
    if direction == 'LR->':
        if inclusive:
            M = config.Mprefixsum1
        else:
            M = config.Mprefixsum0
    elif direction == '<-RL':
        if inclusive:
            M = config.Msuffixsum1
        else:
            M = config.Msuffixsum0
    else:
        return None

    scan = X @ M
    return scan


def propagate(X, direction='both', inclusive=True, mask=None, eps=1.0e-8):
    """
    Accumulate positive values LR-> and/or <-RL, 
    enforcing boundaries (0, 1), applied along the role 
    dimension of X [nbatch x nftr x nrole], with optional masking
    Assumes localist roles and that all elements of 
    X and mask are in (0,1). Implemented with directional scan 
    of fuzzy a OR b = NOT((NOT a) AND (NOT b)) in log domain
    # todo: apply to unrestricted inputs?
    # todo: use logcumsumexp
    """
    assert np.all((0.0 <= X.data.numpy()) & (X.data.numpy() <= 1.0)), \
        print('Error in propogate: X out of domain')
    if mask is not None:
        assert np.all((0.0 <= mask.data.numpy()) & (mask.data.numpy() <= 1.0)),\
            print('Error in propogate: mask out of domain')

    Y = apply_mask(X, mask)

    # Apply bidirectionally
    if direction == 'both':
        prop_LR = propagate(Y, 'LR->', inclusive, eps)
        prop_RL = propagate(Y, '<-RL', inclusive, eps)
        return (prop_LR, prop_RL)

    # Apply unidirectionally
    prop = 1.0 - exp(scan(log(1.0 - Y + eps), direction, inclusive))
    return prop


def inhibit(X, direction='both', mask=None, eps=1.0e-8):
    """
    Apply directional inhibition LR-> and/or <-RL along 
    role dimension of X [nbatch x nftr x nrole], with optional masking
    Assumes localist roles and that all elements of 
    X and mask are in (0,1). (This implementation follows 
    the stick-breaking construction of the Dirichlet Process, 
    performed in the log domain.)
    xxx sharpen (polarize) inputs before applying inhibition?
    # todo: apply to unrestricted inputs?
    todo: use logcumsumexp
    """
    assert np.all((0.0 <= X.data.numpy()) & (X.data.numpy() <= 1.0)), \
        print('Error in inhibit: X out of domain')
    if mask is not None:
        assert np.all((0.0 <= mask.data.numpy()) & (mask.data.numpy() <= 1.0)), \
            print('Error in inhibit: mask out of domain')

    Y = apply_mask(X, mask)

    # Apply bidirectionally
    if direction == 'both':
        val_LR = inhibit(Y, 'LR->', mask)
        val_RL = inhibit(Y, '<-RL', mask)
        return (val_LR, val_RL)

    # Apply unidirectionally
    inhib = scan(log(1.0 - Y + eps), direction, inclusive=False)
    val = exp(log(Y) + inhib)
    return val


#    @staticmethod
#    def test():
#        inhib = DirectionalInhibition()
#        X = 0.01 * torch.ones((1, 2, config.nrole))
#        X[0,0,0] = X[0,0,2] = 0.99
#        X[0,1,1] = X[0,1,3] = 0.99
#        mask = 0.99 * torch.ones((1,config.nrole))
#        mask[0,5:] = 0.01
#        Y_LR, Y_RL = inhib(X,mask)
#        print(f'Y_LR = {np_round(Y_LR[0])}')
#        print(f'Y_RL = {np_round(Y_RL[0])}')
