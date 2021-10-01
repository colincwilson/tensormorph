# -*- coding: utf-8 -*-

# Record and report internal representations and processing

import re, sys
import numpy as np
import pandas as pd
import torch
import config
from morph import Morph


class Recorder:
    """
    Store info posted by grammar submodules, 
    dump and write to file as requested
    """

    def __init__(self):
        self.record = {}

    # xxx deprecated
    def set_values(self, prefix, keyvals):
        """ Record (key,value) pairs """
        for x, y in keyvals.items():
            self.record[prefix + '-' + x] = y.detach()

    # xxx deprecated
    def update_values(self, prefix, keyvals):
        """ Update record of (key,value) pairs """
        for x, y in keyvals.items():
            x = prefix + '-' + x
            if not x in self.record:
                self.record[x] = [
                    y.detach(),
                ]
            else:
                self.record[x].append(y.detach())

    # xxx deprecated
    # xxx if kept, save tensors with pathlib paths
    def dump(self, save=False):
        """
        Create record object, optionally write to file
        """
        for x, y in self.record.items():
            if isinstance(y, list):
                y = [yi.unsqueeze(1) for yi in y]
                self.record[x] = torch.cat(y, dim=1)
        if save:
            # Save all recorded objects
            # xxx replace with pickling?
            for x, y in self.record.items():
                y = np.clip(y.data.numpy(), -1.0e5, 1.0e5)
                np.save(config.save_dir + '/' + x + '.npy', y)
            # Write filler, role, unbinding matrices
            np.save(config.save_dir + '/filler_matrix.npy', config.F)
            np.save(config.save_dir + '/role_matrix.npy', config.R)
            np.save(config.save_dir + '/unbind_matrix.npy', config.U)
            # Write symbols
            syms = np.array(config.syms)
            np.savetxt(config.save_dir + '/symbols.txt', syms, fmt='%s')
            # Write features
            ftrs = np.array(config.ftrs)
            np.savetxt(config.save_dir + '/features.txt', ftrs, fmt='%s')

        return self.record

    def init(self):
        """ Clear record """
        self.record = {}


def assign_node_names(module, memo=None, prefix=''):
    """
    Assign node names to modules of a model, non-recursive
    (similar to torch.nn.module.named_modules())
    note: use 'node' instead of 'name' to avoid clashes
    """
    if memo is None:
        memo = set()
    if module not in memo:
        memo.add(module)
        for name, m in module.named_modules():
            if m is None:
                continue
            name = re.sub('[.]', '-', name)
            m.node = prefix + ('-' if prefix else '') + name
            #assign_node_names(m, memo, m.node)


def report(grammar, batch):
    """
    Record and report processing for first stem in batch
    """
    config.recorder = Recorder()

    cogrammar, decoder = \
        grammar.cogrammar, config.decoder
    stem, output, morphosyn, max_len = \
        batch['stem'], batch['output'], batch['morphosyn'], batch['max_len']
    morphosyn_str = batch['morphosyn_str']
    stem0 = Morph(
        form=stem.form.narrow(0, 0, 1),
        form_str=stem.form_str[0],
        length=stem.length.narrow(0, 0, 1))
    morphosyn0 = [x.narrow(0, 0, 1) for x in morphosyn]
    #morphospec0 = morphospec.narrow(0,0,1)
    #Stem0 = stem.form.narrow(0,0,1); Stem0.slen = Stem.slen.narrow(0,0,1)
    #Morphosyn0 = Morphosyn.narrow(0,0,1)
    #pred = grammar(batch)
    pred = cogrammar(stem0, morphosyn0, max_len)
    pretty_print(cogrammar, decoder, output, morphosyn_str)
    pred2 = grammar(batch)  # xxx hack to see output for phonology-only training
    print(pred2._str()[0])
    if cogrammar.reduplication:
        pretty_print(cogrammar.base_cogrammar, decoder)
        pretty_print(cogrammar.red_cogrammar, decoder)
    #print('tau_affix: '
    #     f'{np.round(config.grammar.cogrammar.cogrammar.affixer.tau.item(), 2)}')
    tau_morph = torch.exp(
        config.morph_attender.tau) + config.morph_attender.tau_min
    tau_posn = torch.exp(
        config.posn_attender.tau) + config.posn_attender.tau_min
    tau_decode = torch.exp(config.decoder.tau) + config.decoder.tau_min
    print(
        f'tau_morph {np.round(tau_morph.item(), 2)} | tau_posn {np.round(tau_posn.item(), 2)} | tau_decode {np.round(tau_decode.item(), 2)}'
    )
    if config.phonology is not None and config.phonology != 0:
        cntxt = torch.ones(1)
        w_faith = cogrammar.phonology.w_faith
        w_nochng = torch.exp(cogrammar.phonology.w_nochng(cntxt)) + w_faith
        w_nochng_min, w_nochng_max = torch.min(w_nochng), torch.max(w_nochng)
        w_nodeln = torch.exp(cogrammar.phonology.w_nodeln(cntxt)) + w_faith
        w_noepen = torch.exp(cogrammar.phonology.w_noepen(cntxt)) + w_faith
        print(
            f'w_nochng [{round(w_nochng_min)}, {round(w_nochng_max)}] | w_nodeln {round(w_nodeln)} | w_noepen {round(w_noepen)}'
        )
    #print(f'tau_morph: {np.round(config.morph_attender.tau.item(), 2)}')
    #print(f'tau_posn: {np.round(config.posn_attender.tau.item(), 2)}')
    #print(f'tau_decode: {np.round(decoder.tau.item(), 2)}')
    #print(f'morph_gate: {np.round(cogrammar.morphology_hwy.item(), 2)}')

    #print('affixer affix_copy: '
    #     f'{cogrammar.affixer.context2affix_copy.bias.cpu().data.numpy()}')
    #print('affixer pivot alpha =',
    #    cogrammar.affixer.pivoter.context2alpha.bias.data.numpy())
    #print('pivoter W0 =', cogrammar.pivoter.W0.weight.data.numpy())
    #print('pivoter bias0 =', cogrammar.pivoter.W0.bias.data.numpy())
    #print('pivoter W1 =', cogrammar.pivoter.W1.weight.data.numpy())
    #print('pivoter bias1 =', cogrammar.pivoter.W1.bias.data.numpy())
    #print('pivoter a =', cogrammar.pivoter.a.data.numpy())
    if cogrammar.correspondence is not None:
        print('alpha_corresp: '
              f'{np.round(cogrammar.correspondence.alpha.cpu().data[0], 4)}')

    config.recorder = None


def pretty_print(cogrammar, decoder, output=None, morphosyn_str=None):
    """
    Report processing for first stem in batch
    """
    node = cogrammar.node
    print('\n**' + node + '**')
    stem = cogrammar.stem
    affix = cogrammar.affix
    pred = cogrammar.output

    stem_str_plain = stem._str(markup=False)[0]
    stem_str = stem._str()[0]
    morphosyn_str = '--' if morphosyn_str is None \
        else morphosyn_str[0]
    affix_str = '--' if affix is None \
        else affix._str()[0]
    output_str = '--' if output is None \
        else output._str()[0]
    pred_str = pred._str()[0]

    print(
        f'🌱 {stem_str_plain}    ⛈️  {morphosyn_str}    🌲 {output_str}    💡 {pred_str}'
    )
    print(f'stem: {stem_str}')
    print(f'affix: {affix_str}')
    print(f'pivot_stem: {np.round(stem.pivot.cpu().data.numpy(), 2)}')
    print(f'pivot_affix: {np.round(affix.pivot.cpu().data.numpy(), 2)}')
    print(f'copy_stem: {np.round(stem.copy.data.cpu().numpy(), 2)}')
    print(f'copy_affix: {np.round(affix.copy.data.cpu().numpy(), 2)}')
    #print('\tWpos:', Wpos.weight[:,0]) # np.round(Wpos.numpy(), 2))
    #print('\tWneg:', Wneg.weight[:,0]) # np.round(Wneg.numpy(), 2))
    #print('copy_affix:', np.round(copy_affix.data.numpy(), 2))
    #print('unpivot:', np.round(pivot_affix.data.numpy(), 2))

    if 0:  # print affix tpr, etc.
        print(np.round(record['root-Affix'].data[0, :, 1].numpy(), 2))
        print(np.round(record['root-Output'].data[0, :, 0].numpy(), 2))
        print(np.round(record['root-Output'].data[0, :, 1].numpy(), 2))
        pred_prob = config.decoder(record['root-Output'])
        pred_prob = exp(log_softmax(pred_prob, 1))
        print(np.round(pred_prob.data[0, :, 1].numpy(), 3))

    #print(np.round(record['root-affix_tpr'].data[0,0,:].numpy(), 2))
    #print('morph_indx:', np.round(morph_indx.data.numpy(), 2))
    #if cogrammar.redup:
    #    pretty_print(cogrammar.cogrammar, None, header='**reduplicant**')


def labeled_tensor(M, col_names, row_names):
    """
    Convert 2D tensor to labeled data frame
    xxx todo: also label 1D tensors
    """
    M = pd.DataFrame(M.data.numpy(), columns=col_names, index=row_names)
    return M


def round(x):
    """
    Round and format tensor containing single float
    """
    y = np.round(x.item(), 2)
    y = f'{y:0.02f}'
    return y