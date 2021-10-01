import sys
import torch

sys.path.append('../tensormorph')
import config
import tensormorph
from morph import Morph
from prosodic_parser import ProsodicParser
#from truncater2 import BiTruncater

data_select, reduplication = "chamorro_um", False
#data_select, reduplication = "marcus_abb", True
tensormorph.init(data_select, features='one_hot', reduplication=reduplication)
form_embedder = config.form_embedder
decoder = config.decoder

data_select, reduplication = "chamorro_um", False
#data_select, reduplication = "marcus_abb", True
tensormorph.init(data_select, features='one_hot', reduplication=reduplication)

prosody = ProsodicParser()
#truncater = BiTruncater()
direction = {0: 'LR->', 1: '<-RL'}[1]
for stem_str in ['plana', 'plantas', 'plantasta', 'ants']:
    stem_str = form_embedder.string2delim([stem_str], split=True)
    stem = Morph(form_str=stem_str)
    stem_mark = prosody.show(stem, direction)
    print(stem_str, '->', stem_mark)

    #truncater.show(stem)