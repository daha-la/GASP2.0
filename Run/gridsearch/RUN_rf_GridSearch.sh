#!/usr/bin/env zsh
chemFeat="encodings/acceptors3_features.tsv"
blosum62Amb="encodings/blosum62Amb.tsv"
strucinform="encodings/h26_50sph_blosum.tsv"
gnn_embedding='encodings/progres_Nt.tsv'

n_estimators="400 800 1000 1200 1600"
criterion=" gini entropy "
max_features=" sqrt log2 "
max_depth="20 50 100"
min_samples_split=" 2 5 10 "
min_samples_leaf=" 1 2 4 "
eval_metric='auc-roc'
cv_search=5
outparam=../../Results/hyperparam_tuning/best_params/progres_Nt_rf_cv${cv_search}_${eval_metric}
cv_results=../../Results/hyperparam_tuning/cv_results/progres_Nt_rf_cv${cv_search}_${eval_metric}
random_state=42


python ../src/Model_Architectures/randomforest/GridSearch.py -i enzyme cid -c reaction -a $chemFeat $gnn_embedding -n $n_estimators --crit $criterion -maxD $max_depth -minS $min_samples_split -minL $min_samples_leaf -maxF $max_features --cv_s $cv_search --op $outparam --cv_res $cv_results -rand $random_state < data/trainset_GTP.tsv


