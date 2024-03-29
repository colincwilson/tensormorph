# -*- coding: utf-8 -*-

import config
from recorder import labeled_tensor
from tpr import *
from morph import MorphOp, Morph
from affixer import Affixer
from affixer2 import Affixer2
#from pivoter import BiPivoter
from truncater import BiTruncater
from matcher import Matcher3, EndMatcher3
from phonology import Phonology
from morphosyn_sequencer import MorphosynSequencer
#from phonology import PhonoRules, PhonoRule
#from torch.nn import LayerNorm
#from torch.distributions.continuous_bernoulli import ContinuousBernoulli
#from torch.nn import TransformerEncoderLayer


class MultiCogrammar(nn.Module):
    """
    Apply sequence of affixation operations.
    """

    def __init__(self):  # xxx specify number of cogrammars
        super(MultiCogrammar, self).__init__()
        self.cogrammar = Cogrammar()
        self.reduplication = False
        self.correspondence = None  # xxx not used
        self.nslot = 1
        self.Mslot2dim_attn = \
            Parameter(0.1 * torch.randn(self.nslot * config.ndim))
        #self.sequencer = MorphosynSequencer()
        # xxx experimental
        # xxx add commandline parameter for number of phonological patterns
        if config.phonology > 0:
            self.phonology = Phonology(dcontext=1, npattern=config.phonology)
            self.morphology_hwy = Parameter(torch.randn(1))  # Highway gate
        else:
            self.phonology = None
        #self.phonology = TransformerEncoderLayer(
        #    d_model = config.dsym, nhead = config.dsym // 3,
        #    dim_feedforward = 50, dropout = 0.0, activation = 'gelu')

        #Mslot2dim_attn_ = -0.5 * torch.arange(self.nslot-1, -1, step = -1) \
        #                            .unsqueeze(0) \
        #                            .expand(config.ndim, self.nslot)
        #self.Mslot2dim_attn = Parameter(Mslot2dim_attn_)
        # xxx initialize with log(1/nslot) ?
        #self.prior = torch.tensor([0.0, -2.0, -2.0], dtype=torch.float).unsqueeze(0)
        #self.slot_attn = nn.Sequential(
        #    nn.Linear(config.ndim, self.nslot * config.ndim),
        #    nn.Sigmoid() )

    def forward(self, stem, morphosyn, max_len):
        nbatch = stem.form.shape[0]
        dim2embed = config.morphosyn_embedder.dim2embed
        morphosyn_zeros = [dim2embed[dim][:, 0] for dim in dim2embed.keys()]
        #morphospec = [(morphosyn[j][:,0] != 1).float()
        #              for j in range(config.ndim)]
        #morphospec = torch.stack(morphospec, -1)
        #print(morphospec.shape)
        Mslot2dim_attn = self.Mslot2dim_attn \
                            .view(config.ndim, self.nslot)
        Mslot2dim_attn = torch.sigmoid(Mslot2dim_attn)
        #Mslot2dim_attn = softmax(Mslot2dim_attn, dim = 0)
        #Mslot2dim_attn = self.sequencer(morphosyn, self.nslot)
        #Mslot2dim_attn = self.slot_attn(morphospec) \
        #                    .view(-1, config.ndim, self.nslot)
        #print(Mslot2dim_attn.shape); sys.exit(0)
        #Mslot2dim_attn = 0.9 * Mslot2dim_attn + 0.1 # Enforced smoothing
        #print(morphospec); sys.exit(0)
        #Mdim2units = config.morphosyn_embedder.Mdim2units
        #Mdim2unspec = config.morphosyn_embedder.Mdim2unspec
        #self.cogrammar.affixer.reset(nbatch)

        stem_i = stem
        for i in range(self.nslot):
            dim_attn_i = Mslot2dim_attn[..., i]
            morphosyn_i = [
                dim_attn_i[j] * morphosyn[j]  #+  
                #    #((1.0 - dim_attn_i[j]) * morphosyn_zeros[j]).view(1,-1)
                for j in range(config.ndim)
            ]
            #morphosyn_i = morphosyn
            output_i = self.cogrammar(stem_i, morphosyn_i, max_len)

            #if config.recorder is not None:
            #    self.trace['stem_str'] = self.cogrammar.stem._str()[0]
            #    self.trace['affix_str'] = self.cogrammar.affix._str()[0]
            #    self.trace['output_str'] = self.cogrammar.output._str()[0]
            stem_i = output_i

        output = output_i

        # Apply phonology to output of morphology [experimental]
        if self.phonology is not None:
            phon_form = morph_form = output.form
            # Phonology applies persistently (2x)
            for i in range(1):
                phon_form = self.phonology(phon_form, torch.zeros(nbatch, 1))
            # Highway connection from output of morphology
            morphology_hwy = sigmoid(self.morphology_hwy)  # xxx bias?
            #output.form = morphology_hwy * morph_form + \
            #              (1.0 - morphology_hwy) * phon_form
            output.form = phon_form  # xxx deactivate morph hwy gate

        # xxx use trace dict
        if config.recorder is not None:
            self.stem = stem  # self.cogrammar.stem
            self.affix = self.cogrammar.affix
            self.output = output  # xxx self.cogrammar.output
            #for key, val in self.trace.items():
            #    print(key, val)
            print(f'Mslot2dim_attn')
            print(
                labeled_tensor(Mslot2dim_attn,
                               [f'slot{i}' for i in range(self.nslot)],
                               config.morphosyn_embedder.dims))
            #print(f'alpha0 {alpha0.data[:,0,0]}, alpha1 {alpha1.data[:,0,0]}')
            #np.save(config.save_dir+'/output0.npy', output0.form.data.numpy())
            #print(f'Winhib = {torch.exp(self.sequencer.W)}')
            #self.w_morph = self.w_morph * .90 # anneal phonology

        return output


class Cogrammar(nn.Module):
    """
    Apply single affixation operation.
    """

    def __init__(self):
        super(Cogrammar, self).__init__()
        self.affixer = Affixer()  # Affixer2()
        self.morph_op = MorphOp()
        self.reduplication = False
        self.correspondence = None  # xxx not used
        # Phonological conditioning of morphology
        if config.dmorphophon > 1:
            self.morphophon1 = nn.LSTM(
                input_size=config.dsym,
                hidden_size=config.dmorphophon,
                num_layers=1,
                batch_first=True,
                bidirectional=False)
            #self.morphophon1 = EndMatcher3(
            #    dcontext = 1,
            #    nfeature = config.dsym,
            #    npattern = 50)
            #self.morphophon2 = nn.Sequential(
            #    nn.Linear(50, 2*config.dmorphophon),
            #    nn.Tanh() )
            #self.
            #self.w_mphon = 1.0/float(2.0 * config.dmorphophon)
        else:
            self.morphophon1 = None
            self.w_mphon = 1.0

    def forward(self, stem, morphosyn, max_len):
        """
        Map stem to affixed output
        todo: apply truncation before affixation
        """

        # Morphophonology of stem
        nbatch = stem.form.shape[0]
        if self.morphophon1 is not None:
            x, (h_n, c_n) = self.morphophon1(stem.form.transpose(1, 2))
            mphon = h_n[0]
            #mphon = self.morphophon1(stem.form, torch.zeros(nbatch, 1))
            #mphon = self.morphophon2(mphon)
        else:
            mphon = torch.zeros((nbatch, config.dmorphophon),
                                requires_grad=False)

        # Affixation operation
        context = morphosyn + [mphon]
        stem, affix, p_zero = self.affixer(stem, context)
        output = self.morph_op(stem, affix)

        # Zero affixation (similar to highway connection)
        #output.form = (1.0 - p_zero) * output.form + \
        #              p_zero * stem.form

        # xxx use trace dict
        if config.recorder is not None:
            self.stem = stem
            self.affix = affix
            self.output = output
            #if self.morphophon is not None:
            #    print(mphon[0])

        return output
