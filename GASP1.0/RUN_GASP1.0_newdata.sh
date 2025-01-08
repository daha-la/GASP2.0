#!/usr/bin/env zsh

# Script for running the original GASP predictions with new sequences
# The original model is trained with the following settings:
# - Random Forest with 1000 trees
# - Random state set to 42
# - Blosum62Amb encoding
# - Acceptor3 features
# - Trainset from GT-Predict

# The arguments are name of the new dataset, the suffix of the encoding used to encode the sequences,
# as well as whether the dataset is a combination dataset or not. The default is no.
new_data=$1
encoding=$2
combinations=${3:-no}

# Define the paths to the files
chemFeat="../Representation_Strategies/substrate/encodings/acceptors3_features.tsv"
blosum62Amb_newseq="encodings/blosum62Amb_$encoding.tsv.gz"
data="../Data/$new_data.tsv"

# Check if the dataset is a combination dataset. If so, change the path to the data.
if [ $combinations = "yes" ]; then
    data="../Data/combinations/$new_data.tsv"
fi

# test
python ../src/randomforest/randomforest_test.py -m ../results/models/model_blosum62Amb.rf.joblib.gz -a $chemFeat $blosum62Amb_newseq < $data > predictions/pred_blosum62Amb_$new_data.tsv
