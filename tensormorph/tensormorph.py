# -*- coding: utf-8 -*-

import pickle, yaml, sys
from pathlib import Path
import numpy as np
import torch
from pytorch_lightning import Trainer

import config
from phon import config as phon_config
from phon import features as phon_features
from phon import str_util
import morph
import data_util
from form_embedder import FormEmbedder
from morphosyn_embedder import UnimorphEmbedder, DefaultEmbedder, OneHotEmbedder
from decoder import Decoder  #, LocalistAlignmentDecoder
from grammar import Grammar
from mcmc import MCMCSampler
from prior import Prior
import unit_test


def init(args):
    """
    Initialize tensormorph model
    """
    # # # # # # # # # #
    # Load default model options into args
    with open('model_config.yaml', 'r') as f:
        args_default = yaml.load(f)
    for key, val in args_default.items():
        if not hasattr(args, key):
            setattr(args, key, val)

    # Load options into config
    for key, val in vars(args).items():
        setattr(config, key, val)
    if not hasattr(config, 'save_dir'):
        config.save_dir = str(Path.home() / 'Desktop/tmorph_output')
    if not hasattr(config, 'data_name'):
        config.data_name = config.data_pkl
    if config.gpus > 0:
        config.device = torch.device('cuda:0')
    else:
        config.device = torch.device('cpu')
    config.args = args

    # String config
    phon_config.init(args)

    # # # # # # # # # #
    # Load data
    config.fdata = Path(config.data_dir) / args.data_pkl
    with open(config.fdata, 'rb') as f:
        data = pickle.load(f)
    config.data = data
    print(f"\ntrain {len(data['data_train'])} "
          f"| val {len(data['data_val'])} "
          f"| test {len(data['data_test'])} ")
    print(data['data_train'].head())

    # # # # # # # # # #
    # Initialize form embedding
    if (args.features == 'one_hot'):
        feature_matrix = phon_features.one_hot_features(data['segments'],
                                                        data['vowels'])
    else:
        feature_matrix = phon_features.import_features(
            Path('.') / 'features' / args.features,
            data['segments'],
            save_file=config.fdata)
    symbol_params = {'feature_matrix': feature_matrix}
    role_params = {'nrole': data['max_len'] + 2}  # at least +2 for delims
    form_embedder = FormEmbedder(symbol_params, role_params)
    config.form_embedder = form_embedder
    for x in ['syms', 'ftrs', 'ftr_matrix', 'F', 'nsym', 'dsym']:
        setattr(config, x, getattr(form_embedder.symbol_embedder, x))
    for x in ['R', 'U', 'nrole', 'drole']:
        setattr(config, x, getattr(form_embedder.role_embedder, x))

    # # # # # # # # # #
    # Operators over role-local embeddings
    # xxx relocate
    # Lag/delay by one position
    config.Mlag = \
        torch.tensor(np.eye(N = config.nrole, k = 1),
                     requires_grad = False,
                     dtype = torch.float,
                     device = config.device)

    # Lead/advance by one position
    config.Mlead = \
        torch.tensor(np.eye(N = config.nrole, k = -1),
                     requires_grad = False,
                     dtype = torch.float,
                     device = config.device)

    # All-prefix-sum operators
    M = np.ones((config.nrole, config.nrole))
    # Inclusive
    config.Mprefixsum1 = \
        torch.tensor(np.triu(M, k = 0),
                     requires_grad = False,
                     dtype = torch.float,
                     device = config.device)

    # Exclusive
    config.Mprefixsum0 = \
        torch.tensor(np.triu(M, k = 1),
                     requires_grad = False,
                     dtype = torch.float,
                     device = config.device)

    # All-suffix-sum operators
    # Inclusive
    config.Msuffixsum1 = \
        torch.tensor(np.tril(M, k = 0),
                     requires_grad = False,
                     dtype = torch.float,
                     device = config.device)

    # Exclusive
    config.Msuffixsum0 = \
        torch.tensor(np.tril(M, k = -1),
                     requires_grad = False,
                     dtype = torch.float,
                     device = config.device)

    # # # # # # # # # #
    # Initialize morphosyn embedding and morphophon conditioning
    if config.morphosyn == 'unimorph':
        config.morphosyn_embedder = UnimorphEmbedder()
    else:
        config.morphosyn_embedder = DefaultEmbedder()
    config.ndim = config.morphosyn_embedder.ndim
    config.dmorphosyn = config.morphosyn_embedder.dmorphosyn
    config.dcontext = config.dmorphosyn + config.dmorphophon  # 2*
    config.context_size = config.morphosyn_embedder.dim2size
    if config.dmorphophon is not None:
        config.context_size += [2 * config.dmorphophon]
    else:
        config.context_size += [1]

    # # # # # # # ## # #
    # Embed data
    config.data_train = data_util.MorphDataset(data['data_train'])
    config.data_val = data_util.MorphDataset(data['data_val'])
    config.data_test = data_util.MorphDataset(data['data_test'])  # xxx postpone

    # # # # # # # # # #
    # Create grammar
    config.grammar = Grammar(
        learning_rate=config.learn_rate,
        reduplication=args.reduplication)  # xxx config

    config.prior = Prior()

    # # # # # # # # # #
    # Run unit tests
    #unit_test.run(); sys.exit(0)
    #print('done')
    return config


def train_and_evaluate():
    """
    Train grammar and report train | test results
    """
    grammar = config.grammar
    trainer = Trainer(
        logger=False,
        progress_bar_refresh_rate=1,
        min_epochs=config.min_epochs,
        max_epochs=config.max_epochs,
        gradient_clip_val=config.grad_clip,
        stochastic_weight_avg=config.stochastic_weight_avg,
        auto_lr_find=False,  # finds values wayyyyy too small
        #precision=16, use_amp=False,
        #num_processes = 4, # number of cpus xxx make config option
        gpus=config.gpus,
    )
    trainer.fit(grammar)

    with open(Path(config.save_dir) / f'{config.data_name}_config.pkl',
              'wb') as f:
        pickle.dump(config.args, f)
    torch.save(grammar.state_dict(),
               Path(config.save_dir) / f'{config.data_name}_model.pt')
    evaluate('train')
    evaluate('val')
    evaluate('test')


def evaluate(split):  # xxx move testing to grammar module
    data = config.data[f'data_{split}']  # raw data
    data_embed = getattr(config, f'data_{split}')
    batch = data_util.morph_batcher(data_embed)

    #config.decoder.add_noise = False
    pred = config.grammar(batch)
    pred = pred._str()
    data['pred'] = [str_util.remove_delim(x) for x in pred]
    data['score'] = [int(i) for i in (data['pred'] == data['output'])]
    pred_accuracy = data['score'].mean()
    pred_errors = data[(data['score'] == 0)]
    print(f'{split} accuracy: {pred_accuracy} '
          f'({len(pred_errors)}/{len(data)} errors)')
    if len(pred_errors) > 0:
        print(pred_errors.head())
    data.to_csv(
        Path(config.save_dir) / f'{config.data_name}_{split}_results.csv',
        index=False)

    #config.decoder.add_noise = True


#def sample_analysis():
#    sampler = MCMCSampler()
#    sampler.sample()
