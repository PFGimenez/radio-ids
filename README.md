# Data pipelines

There are two pipelines : micro and macro

## Data

The data are waterfalls measured with a HackRF device.

_merge_raspi_ merges the sources into one source using with the _max_ aggregator. This unique source is used by the micro, macro and extractors models.

# Pipeline for micro model

_learn_extract_micro_ learns a CNN autoencoder from data

_extract_micro_ extracts features from data using the CNN autoencoder

_learn_model_micro_ learns a predictive model from the features

# Pipeline for macro model

_extract_macro_stage_1_ extract features from data (features are hand-chosen)

_extract_macro_stage_2_ extract the features using PCA

_learn_model_macro_ learns a predictive model from the features

# Pipeline for extractors model

No specific pipeline : the models are learnt with _learn_extract_micro_.

# Pipeline for multisource model

_learn_multisource_ learns a model to detect anomaly from multisource

# Evaluation

_evaluate_ evaluates the models (micro, macro, extractors) on test data and outputs the results
