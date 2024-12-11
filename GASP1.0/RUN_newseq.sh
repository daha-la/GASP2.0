#!/usr/bin/env zsh
new_seqs=$1
encoding=$2
chemFeat="../encodings/acceptors3_features.tsv"
blosum62Amb_newseq="../encodings/blosum62Amb_$encoding.tsv.gz"

python ../randomforest/randomforest_test.py -m ../results/models/model_blosum62Amb.rf.joblib.gz -a $chemFeat $blosum62Amb_newseq < combinations/$new_seqs.tsv > predictions/pred_blosum62Amb_$new_seqs.tsv
