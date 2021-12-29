## Prepare data:

python 01_prepare_data.py --config <data>.yaml

where

<data>.yaml is the _full_ path to a data config file (e.g., ../tensormorph_data/chamorro/chamorro_um.yaml)

Ex. python 01_prepare_data.py --config ../tensormorph_data/french/french_verbs.yaml

## Train and evaluate model:

python 02_train_model.py --data <data_dir>/<data_subset>

where

<data_dir> is by default a subdirectory of ../tensormorph_data

<data_dir>/<data_subset>.yaml specifies data and model configs

Ex. python 02_train_model.py --data french/french_verbs --morphosyn unimorph
