#!/usr/bin/env zsh

# Script for running the original GASP predictions with new sequences
# The original model is trained with the following settings:
# - Random Forest with 1000 trees
# - Random state set to 42
# - Blosum62Amb encoding
# - Acceptor3 features
# - Trainset from GT-Predict

# The arguments are name of the file containing the new sequences and the suffics of the encoding used to encode the sequences
new_seqs=$1
encoding=$2
chemFeat="../Representation_Strategies/substrate/encodings/acceptors3_features.tsv"
blosum62Amb_newseq="encodings/blosum62Amb_$encoding.tsv.gz"

# test
python ../randomforest/randomforest_test.py -m ../results/models/model_blosum62Amb.rf.joblib.gz -a $chemFeat $blosum62Amb_newseq < combinations/$new_seqs.tsv > predictions/pred_blosum62Amb_$new_seqs.tsv
