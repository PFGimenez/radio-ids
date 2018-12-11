# Data pipelines

There are two pipelines : micro and macro

## Data

The data are waterfalls measured with a HackRF device.

# Pipeline for micro model

_learn_extract_micro_ learns a CNN autoencoder from data

_extract_micro_ extracts features from data using the CNN autoencoder

_learn_model_micro_ learns a predictive model from the features

_evaluate_micro_ evaluates the model on test data and outputs the results

# Pipeline for macro model

_extract_macro_stage_1_ extract features from data (features are hand-chosen)

_learn_extract_macro_stage_2_ learns the PCA from data

_extract_macro_stage_2_ extract the features using the PCA

_learn_model_macro_ learns a predictive model from the features

_evalute_macro_ evaluates the model on test data and outputs the results
