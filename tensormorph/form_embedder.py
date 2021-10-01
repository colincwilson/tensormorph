# -*- coding: utf-8 -*-

import re, sys
from pathlib import Path
import config
from tpr import *
from symbol_embedder import SymbolEmbedder
from role_embedder import RoleEmbedder

from phon import str_util


class FormEmbedder():
    """"
    Map strings to/from symbol id tensors and TPR tensors.
    """

    def __init__(self, symbol_params, role_params):
        self.symbol_embedder = SymbolEmbedder(**symbol_params)
        self.role_embedder = RoleEmbedder(**role_params)
        self.syms = self.symbol_embedder.syms
        self.sym2id = \
            { sym:i for i,sym in enumerate(self.syms) }
        self.id2sym = \
            { i:sym for sym,i in self.sym2id.items() }
        #print(self.sym2id); print(self.id2sym)

    def sym2vec(self, x):
        """
        Symbol to filler vector
        """
        F = config.F
        return F.data[:, self.sym2id[x]]

    def string2vec(self, x, delim=True):
        """
        Space-separated string to matrix of filler vectors
        """
        F = config.F
        idx, lens = self.string2idvec(x, delim)
        return F[:, idx], lens

    # # # # # Strings to tensors # # # # #

    def string2idvec(self, x, delim=True, pad=False):
        """
        Space-separated string to vector of symbol ids, optionally with 
        begin/end delimiters added and zero-padding at end; also returns unpadded string length.
        """
        # Recursively apply to each member of batch
        # xxx create tensor batch here
        if isinstance(x, list):
            return [self.string2idvec(xi, delim, pad) for xi in x]
        # Optionally add begin/end delimiters
        if delim:
            x = str_util.add_delim(x)
            #x = self.string2delim(x)
        # Convert symbols to indices
        sym = x.split(' ')
        sym_id = [self.sym2id[xi] if xi in self.sym2id else None for xi in sym]
        # Report missing symbols and fail fast
        if None in sym_id:
            print(f'string2idvec error: {x}')
            print([sym_id for sym_id in zip(sym, sym_id)])
            sys.exit(0)
        # Optionally pad with trailing zeros
        if pad:
            sym_id = sym_id + \
                [0,] * (config.nrole - len(sym))
        y = torch.tensor(
            sym_id, dtype=torch.long, device=config.device, requires_grad=False)
        # Unpadded length
        y_len = torch.tensor([len(sym)],
                             dtype=torch.long,
                             device=config.device,
                             requires_grad=False)
        return y, y_len

    def string2tpr(self, x, delim=True):
        """
        Space-separated string to TPR in real (pre-tanh) domain
        """
        # Recursively apply to each member of batch
        # xxx create tensor batch here
        if isinstance(x, list):
            return [self.string2tpr(xi, delim) for xi in x]
        # Convert string to symbol ids
        y, y_len = self.string2idvec(x, delim)
        # Report lengths greater than number of roles and bail out
        if y_len > config.nrole:
            print(
                f'string2tpr error: length of string (= {y_len.item()}) longer than nrole (= {config.nrole}) for input: {x}'
            )
            sys.exit(0)
        # Convert to matrix
        Y = torch.zeros((config.dsym, config.drole),
                        device=config.device,
                        requires_grad=False)
        for i in range(y_len):
            Y += torch.ger(config.F[:, y[i]], config.R[:, i])  # outer product
        # note: do not pad end with delim, observationally this results in
        # a pathological preference for long suffixes with many epsilons;
        # instead pad with epsilons -- or wildcards?
        #for i in range(y_len, config.nrole):
        #    Y += torch.ger(config.F[:,2], config.R[:,i])

        # Map from [-1,+1] into real (pre-tanh) domain,
        # with discreteness approximation tanh(3.0) ≈ 0.995, tanh(5.0) ≈ .9999
        Y = 3.0 * Y

        return Y

    # # # # # Tensors to strings # # # # #

    def idvec2string(self, form_id, **markup):
        """
        Map sequence of symbol ids to string, optionally adding markup and trimming
        """
        if isinstance(form_id, torch.Tensor):
            form_id = form_id.detach().cpu().numpy()
        # Convert symbol id sequences to symbol strings
        syms = np.array(self.syms)[form_id]
        nbatch = syms.shape[0]
        y = [' '.join(syms[i]) for i in range(nbatch)]

        # Optionally add copy/delete and pivot markup
        copy_thresh = 0.5
        pivot_thresh = 0.25
        if 'copy' in markup and markup['copy'] is not None:
            copy = markup['copy'].data.numpy()
            for i in range(nbatch):
                syms_i = [
                    f'⟨{sym}⟩' if copy[i][j] < copy_thresh else sym
                    for j, sym in enumerate(y[i].split(' '))
                ]
                y[i] = ' '.join(syms_i)
        if 'pivot' in markup and markup['pivot'] is not None:
            pivot = markup['pivot'].data.numpy()
            for i in range(nbatch):  # • or ●
                syms_i = [
                    f'{sym} •' if pivot[i][j] > pivot_thresh else sym
                    for j, sym in enumerate(y[i].split(' '))
                ]
                y[i] = ' '.join(syms_i)

        # Optionally delete all symbols after first end delimiter
        # and trailing epsilon + markup sequences
        if 'trim' in markup:
            for i in range(nbatch):
                y[i] = re.sub(f'({config.eos}[^ ]*).*', \
                              '\\1', y[i])
                y[i] = re.sub(f'⟨?{config.epsilon}[{config.epsilon}⟨⟩• ]*$', \
                              '', y[i])

        return y

    def tpr2string(self, X, **markup):
        """
        Map form X to string by maximum-likelihood decoding at each position
        """
        sim = config.decoder(X)  # [nbatch x nsym x nrole]
        form_id = sim.argmax(1)  # [nbatch x nrole]
        y = self.idvec2string(form_id, **markup)
        return y