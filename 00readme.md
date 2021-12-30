## Prepare data:

python 01_prepare_data.py --config <data_model_config>.yaml

where <data_model_config>.yaml is the \_full\* path to a data+model config file

Ex. python 01_prepare_data.py --config ../tensormorph_data/chamorro/chamorro_um.yaml

Ex. python 01_prepare_data.py --config ../tensormorph_data/french/french_verbs.yaml

## Train and evaluate model:

python 02_train_model.py --config <data_model_config>.yaml

where <data_model_config>.yaml is the \_full\* path to a data+model config file, as above

Ex. python 02_train_model.py # runs Chamorro -um- simulation by default

Ex. python 02_train_model.py --config ../tensormorph_data/french/french_verbs.yaml
