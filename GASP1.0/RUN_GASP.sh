#!/usr/bin/env zsh
chemFeat="../encodings/acceptors3_features.tsv"
blosum62Amb="../encodings/blosum62Amb.tsv.gz"
strucinform="../encodings/h26_25sph.tsv "

n_estimators=" 1600"
criterion=" entropy "
max_features=" log2 "
#max_depth=
min_samples_split=" 2 "
min_samples_leaf=" 1 "

random_state=42


# train with original settings
python ../randomforest/randomforest_train.py -i enzyme cid -c reaction -n 1000 -o ../results/models/model_blosum62Amb.rf.joblib.gz -a $chemFeat $blosum62Amb -rand $random_state < ../data/trainset_GTP.tsv

# train with custom settings
#python randomforest/randomforest_train.py -i enzyme cid -c reaction -n $n_estimators -crit $criterion -maxF $max_features -minS $min_samples_split -minL $min_samples_leaf -o results/models/model_h26_25sph.rf.joblib.gz -a $chemFeat $strucinform -rand $random_state < data/trainset_GTP.tsv

# test
python ../randomforest/randomforest_test.py -m ../results/models/model_blosum62Amb.rf.joblib.gz -a $chemFeat $blosum62Amb < ../data/testset_D1.tsv > ../results/predictions/pred_blosum62Amb.tsv

