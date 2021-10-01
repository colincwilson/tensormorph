- Consistent naming of splits: val ~ dev

- Early stopping (dev loss ~ 0.0) in training loop

- Study precisons of morph attention, posn attention, and decoder
    - Empirical max values after successful training are ~ 1.5 on log scale:
        tau_morph: 1.5
        tau_posn: 1.5
        tau_decode: 1.5 - 2.0
      with additive tau_min set to 0.5
    - Prove that values larger than exp(1.5) + 0.5 ~ 5.0, or exp(2) + 0.5 ~ 8.0, are not necessary for perfect decoding in hand-written models
    - Better control of starting values -- do not want high degrees of precision during initial phase of learning
    - Separate minimum tau parameters for morph and position

- Reimplement stem truncater / stem extracter (e.g., for Romance verb conjugation)
    - esp. edge-based truncation

+ Allow log-domain epsilon masking when computing RBF-Softmax distributions in MorphOp -- otherwise unfilled roles contribute 0s to softmax sum over roles
    - only sensibly applies to stems

- Introduce ndecode parameter (ndecode > nrole) to allow for epsilon (non-writing) steps in decoding that would make the total number of steps greater than nrole (as can happen in partial reduplication)
    - important: decoding steps are not necessarily writing steps!

- Consolidate code here from Dropbox/TensorMorph/00python|00notes

+ Add entry code with rich argparser for quick set-up in demo notebooks
    => entry class src/tensmorph.py

+ Add context sensitivity to pivoter

+ Add Morph class with fields:
    form (TPR representation)
    morph id (minimally stem vs. affix)
    correspondence indices (see below)
    - reimplement all classes to operate on Morphs rather than forms

+ Restore phonological rule functionality [work-in-progress]
    - maybe add id map to training data to distinguish phonology vs. morphology

+ Implement Lexical Phonology stack of layers
    - partial, see MultiCogrammar

+ Add correspondence indices
    correspondence.py

- Handle correspondence indices in affixer
    - Use random indices for affixes?
    - see van Oostendorp on 'colored containment'

+ Transpose attention vector in correspondence module

+ Implement re-copying and back-copying with transformer-like self-attention

+ Make reduplication examples with stem-initial voicing and final a-raising
        with and without recopying

+ Rename affixer => cogrammar

+ Move stem scanning and pivot finding into Affixer modoule
    Affixer module now scans stem for morphophonology, 
    locates pivot in stem, and retrieves affix form / copy / unpivot 
    from morphosyn (+ morphophon) context.
    - Should multiple cogrammars be able to share an affixer?

+ Replace vocab_inserter with affixer

+ Make cogrammar subclass for reduplication

+ Make pytorch-lightning wrapper
  (alternatives: catalyst, ignite, skorch, torchbearer)
    => See grammar.py

+ Standardize length-masking implementations.
    Compare with sequence_mask() in
    [https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/misc.py]
    as used in
    [https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/global_attention.py]
        <snippet>
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(~mask, -float('inf'))
        </snippet>
    See also Rush (2018), Annotated Transformer, bottom of p. 54
    => See tpr.epsilon_mask, tpr.begin_mask, tpr.end_mask

- Add default initializations
    Matcher: Wpos, Wneg should have positive defaults (i.e., conjunctive default)
    Pivoting: no pivoting
    Affix: all-epsilon affix
    Stem modification: no deletion

- Review logic of stem_modifier (deleter, trimmer)
    Replace max/min with addition in log domain
    More generally, do operations in log domain as much as possible (see matcher)

- Make evaluator.py compatible with pytorch-lightning wrapper
    now in src/zzz/evaluator.py

- Add epoch-stamping to recorder (videos!)

+ Replace torch.matmul with @ infix operator throughout for readability

+ Standardize naming of precision parameters: c => tau

+ Replaced ad-hoc node naming with pytorch internal names
  (see for example torch.nn.module.named_modules())
  => Node name assignment by recoder.assign_node_names()

+[partial] Within each module, replace recorder-related code  -- which currently 
    distracts from logical flow, with a custom forward hook that is 
    registered on module creation (and that sends designated tensors to 
    the recorder module)
    But: would still need a self attribute _record_ that specifies tensors to be stored and their names -- see bottom of affixer.forward() for example

+ Relocate all datasets from Dropbox to separate tensormorphy_data repo
    convert English redup examples to IPA
