# -*- coding: utf-8 -*-

import re, sys
import pywrapfst, pynini
from pywrapfst import Fst
sys.path.append('../../fst')
from fst import fst_config, fst
from fst.fst import FST, Transition 

epsilon = 'ε'
marker = '•'
wildcard = '□'
segments = ['t', 'r', 'i', 's', 'u', 'm']

stem_str = '>• t r• i s t i• <'
stem = fst.linear_acceptor(stem_str)
T = stem.T.copy()
for t in T:
    if re.search(marker, t.olabel):
        for segment in segments:
            stem.T.add(Transition(
                src=t.dest, ilabel=epsilon, olabel=segment, dest=t.dest))
for t in stem.T:
    t.ilabel = re.sub(marker, '', t.ilabel)
    t.olabel = re.sub(marker, '', t.olabel)
print(stem)
fst.draw(stem, 'stem.dot')

output_str = '> t r u m i s t i <'
output = fst.linear_acceptor(output_str)
print(output)

align = fst.intersect(stem, output)
print(align)
fst.draw(align, 'align.dot')

#stem = '>tristi<'
#fst_stem = pynini.acceptor(stem, token_type="utf8")
#print(fst_stem.print())
#print(fst_stem.input_symbols())
