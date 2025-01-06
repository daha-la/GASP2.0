#!/usr/bin/env zsh

# Script for running the original GASP pipeline
# The original pipeline is run with the following settings:
# - Random Forest with 1000 trees
# - Random state set to 42
# - Blosum62Amb encoding
# - Acceptor3 features
# - Trainset from GT-Predict
# - Testset from GASP paper

chemFeat="../Representation_Strategies/substrate/encodings/acceptors3_features.tsv"
blosum62Amb="encodings/blosum62Amb.tsv.gz"

random_state=42

# train with original settings
python ../Model_Architectures/randomforest/randomforest_train.py -i enzyme cid -c reaction -n 1000 -o results/models/model_blosum62Amb.rf.joblib.gz -a $chemFeat $blosum62Amb -rand $random_state < ../Data/trainset_GTP.tsv

# test
python ../Model_Architectures/randomforest/randomforest_test.py -m results/models/model_blosum62Amb.rf.joblib.gz -a $chemFeat $blosum62Amb < ../Data/testset_D1.tsv > results/predictions/pred_blosum62Amb.tsv

