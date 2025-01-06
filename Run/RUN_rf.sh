#!/usr/bin/env zsh

# Script for running the GASP2.0 pipeline
# The GASP2.0 pipeline requires the specification of the following:
# ChemFeat: Path to the acceptor features file
# Encoding: Path to the protein encoding file
# Identifier: Identifier for the model and predictions
# Model Parameters: Parameters for the Random Forest model
# - n_estimators: Number of trees in the forest
# - criterion: The function to measure the quality of a split
# - max_features: The number of features to consider when looking for the best split
# - max_depth: The maximum depth of the tree
# - min_samples_split: The minimum number of samples required to split an internal node
# - min_samples_leaf: The minimum number of samples required to be at a leaf node
# - random_state: The seed used by the random number generator

# Variables
training_data="../Data/trainset_GTP.tsv"
test_data="../Data/testset_D1.tsv"
chemFeat="../Representation_Strategies/substrate/encodings/acceptors3_features.tsv"
encoding="../Representation_Strategies/protein/encodings/blosum62amb_alignment_nterm.tsv"
identifier="blosum62Amb_Nterm_default"

# Model Parameters
n_estimators=" 1000 "
criterion=" gini "
max_features=" sqrt "
max_depth=None
min_samples_split=" 2 "
min_samples_leaf=" 1 "
random_state=42

# train
python ../Model_Architectures/randomforest/randomforest_train.py -i enzyme cid -c reaction -n 1000 -n $n_estimators -crit $criterion -maxD $max_depth -minS $min_samples_split -minL $min_samples_leaf -maxF $max_features -o ../Results/models/model_$identifier.rf.joblib.gz -a $chemFeat $encoding -rand $random_state < $training_data

# test
python ../Model_Architectures/randomforest/randomforest_test.py -m ../Results/models/model_$identifier.rf.joblib.gz -a $chemFeat $encoding < $test_data > ../Results/predictions/pred_$identifier.tsv

