#!/usr/bin/env python3
import argparse
from argparse import RawTextHelpFormatter
import sys
import math
import numpy as np
import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as AUC
import joblib
import logging as log
import re
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    ndcg_score,
    make_scorer
)

def get_parser():
    parser = argparse.ArgumentParser(description="""
    Conduct a GridSearch on a Random Forest classifier. 
    Reads a tab-separated table with header from stdin and writes prediction table to stdout, 
    if testing. Infile columns:
        (-c) A class column where 1 is for a positive and 0 for a negative observation. 
        (-F) Feature columns, either here on using -a/--annotate.
        (-i) Optionally id column(s) that can map to features from (-a).
    """, formatter_class=RawTextHelpFormatter) # RawTextHelpFormatter to keep newlines.
    
    parser.add_argument("-i", "--id", dest="ids", nargs="+", default=[],
        help="Name of columns that are neither feature nor class so should be identifier\ncolumns. They are simply copied to output for identification.")
    parser.add_argument("-a", "--annotate", nargs="+",
        help="Filename(s) for providing features, that will annotate the infile by matching on overlapping ids given by -i/--id.")
    parser.add_argument("-m", "--model",
        help="Read model (e.g. randomforest.joblib) from this arg and test instead of\ntrain from stdin.")
    parser.add_argument("-n", "--estimators", dest="n_estimators", type=int, default=100, nargs='+',
        help="Number of trees in the forest.")
    parser.add_argument("--crit", dest="criterion", default="gini", nargs='+',
        help="Randomforest criterion for split quality.", choices=["gini", "entropy"])
    parser.add_argument("-maxD", "--max_depth", dest="max_depth", type=int, default=None, nargs='+',
        help="The maximum depth of the tree.")
    parser.add_argument("-minS", "--min_split", dest="min_samples_split", type=int, default=2, nargs='+',
        help="The minimum number of samples required to split an internal node.")
    parser.add_argument("-minL", "--min_leaf", dest="min_samples_leaf", type=int, default=1, nargs='+',
        help="The minimum number of samples required to be at a leaf node.")
    parser.add_argument("-maxF", "--max_feat", dest="max_features", default="sqrt", nargs='+',
        help="The number of features to consider when looking for the best split.")
    parser.add_argument("--class-weight",
        help="Randomforest class weight. Default=equal weight to each datapoint.", choices=["balanced", "balanced_subsample"])
    # uppercase Class since class is a reserved word
    parser.add_argument("-c", "--class", dest="Class",
        help="Name of column with class labels in training data. Default=class.", default="class")
    parser.add_argument("--log",
        help="Logging is sent to stderr. Set flag to a filename to also write to file.")
    parser.add_argument("--cv_s", dest="cv_search", type=int, default=5,
        help="Do k-fold cross-validation for GridSearch, provide k.")
    parser.add_argument("--op","--out_param", dest="out_param", type=str,
        help="Name for the text file with the best parameters.")
    parser.add_argument("-em","--eval_metric", dest="metric", type=str, default='auc-roc',
        help="Metric score for evaluating the best model parameters.", choices=["precision", "recall", "accuracy", "balanced-accuracy", "mcc", "f1", "auc-roc"])
    parser.add_argument("--cv_res","--cv_results", dest="cv_results", type=str,
        help="Name for the tsv file with the cross-validation results from the search.")
    parser.add_argument("-rand", "--random_state", dest="random_state",
        help="Defines the random state of the RandomForestClassifier", default=42)
    
    return parser

def param_into_list(param):
    if type(param) != list:
        return [param]
    else:
        return param
    
def param_grid_gen(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_features):
    """
    Construct parameter grid for GridSearch
    :param n_estimators: number of trees
    :param criterion: criterion for split quality
    :param max_depth: maximum depth of trees
    :param min_samples_split: minimum sample required for split
    :param min_samples_leaf: minimum sample in leaf
    :param max_features: number of features considered when looking for best split
    :param bootstrap: whether trees are bootstrapped or built on entire dataset
    :return: dict of parameter options
    """
    n_estimators = param_into_list(n_estimators)
    criterion = param_into_list(criterion)
    max_depth = param_into_list(max_depth)
    min_samples_split = param_into_list(min_samples_split)
    min_samples_leaf = param_into_list(min_samples_leaf)
    max_features = param_into_list(max_features)
    if max_depth != [None]:
        max_depth.append(None)
    #max_features.append(None)
    param_grid = {"n_estimators": n_estimators,
                  "criterion": criterion,
                  "max_features": max_features,
                  "max_depth": max_depth,
                  "min_samples_split": min_samples_split,
                  "min_samples_leaf": min_samples_leaf}
    #pprint(param_grid)
    return param_grid

def grid_search(train, Class, param_grid, cv_search, metric,random_state):
    """
    Conducting GridSearch
    :param train: training data
    :param Class: name of Class column in training data
    :param param_grid: dict of parameter options
    :param cv_search: k for crossvalidation
    :param metric: evaluation metric for search
    :return: trained best model, dict of best parameter options, and dataframe with cross-validation results
    """
    metric_dict = {"precision": precision_score,
                   "recall": recall_score,
                   "accuracy": accuracy_score,
                   "balanced-accuracy": balanced_accuracy_score,
                   "mcc": matthews_corrcoef,
                   "f1": f1_score,
                   "auc-roc": roc_auc_score,
                   "ndcg": make_scorer(ndcg_score,k=10)
                  }
    eval_metric = metric_dict[metric]
    scorer = make_scorer(eval_metric)
    estimator = RandomForestClassifier(random_state=random_state)
    gridsearch = GridSearchCV(estimator = estimator, param_grid = param_grid, cv = cv_search, verbose=0, n_jobs = -1, scoring=scorer)
    gridsearch.fit(train.drop(columns=Class), train[Class])
    best_params = gridsearch.best_params_
    cv_results = gridsearch.cv_results_
    cv_results_df = pd.concat([pd.DataFrame(cv_results["params"]),pd.DataFrame(cv_results["mean_test_score"], columns=[metric]),pd.DataFrame(cv_results["std_test_score"], columns=[metric+'_std'])],axis=1)
    
    return best_params, cv_results_df

def remove_redundant(*dfs, ignore=None):
    """
    Find and remove columns where all values are identical across all given dfs
    :param dfs: pandas dataframes
    :param ignore: names of columns to ignore
    :return: list of strings names for columns that are redundant to use as features
    """
    # skip None
    dfs = [df for df in dfs if df is not None]
    cat = pd.concat(dfs).drop(columns=ignore)
    redundant = cat.columns[np.all(cat == cat.iloc[0, :], axis=0)]
    if len(redundant) > 0:
        log.info("Removed {} out of {} ({:.2f}%) features that never varies".
                 format(len(redundant), len(cat.columns), len(redundant) / len(cat.columns) * 100))
        return [df.drop(columns=redundant) for df in dfs]
    return dfs


def feature_prep(train, ignore=None):
    """
    Prepare features for random forest. This means we make sure to drop string features and redundant features that never vary.
    :param train: pd dataframe
    :param test: pd dataframe
    :param ignore: columns to ignore that are not features
    :return: modified train and test set ready for random forest
    """

    # for now drop string features, but in the future they should maybe be implemented as 1-hot automatically
    train_feature_names = np.setdiff1d(train.columns, ignore)
    train_feature_dtypes = train[train_feature_names].dtypes
    train_nonum = train_feature_names[(train_feature_dtypes != int) & (train_feature_dtypes != float)]
    if len(train_nonum) > 0:
        log.info("Dropping non-number feature(s): " + ','.join(train_nonum))
        train.drop(columns=train_nonum, inplace=True)

    train, = remove_redundant(train, ignore=ignore)
    return train

def run(train, Class, param_grid, cv_search, metric, random_state, ignore=None):
    if ignore is None: ignore = set()
    elif np.isscalar(ignore): ignore = {ignore}
    else: ignore = set(ignore)
    ignore = ignore - {Class} # obviously don't ignore the class
    train = train.drop(columns=ignore, errors='ignore')
    train = feature_prep(train, ignore.union({Class}))
    if train.isna().any().any():
        nBefore = len(train)
        train = train.dropna()
        log.warning(f"NaN records removed for training: {nBefore} -> {len(train)}")
    log.info("Conduct GridSearch.")
    best_params, cv_results_df = grid_search(train, Class, param_grid, cv_search, metric, random_state)
    return best_params, cv_results_df


def main(args):
    if args.log is None: handlers = None
    else: handlers = [log.StreamHandler(), log.FileHandler(args.log)]
    log.basicConfig(format="%(message)s", level=log.INFO, handlers=handlers)
        
    # reading
    
    infile = pd.read_table(args.infile)
    log.info(f"shape = {infile.shape}")
    
    #set random state
    if args.random_state == None:
        random_state = None
    else:
        random_state = int(args.random_state)
    
    # the original keys, before any potential annotations are added
    infileKeys = infile.columns
    for infname in args.annotate:
        log.info(f"Annotating from {infname}")
        right = pd.read_table(infname)
        # merge on keys overlapping between the two, and that are given as ids, if ids are given.
        onKeys = np.intersect1d(infileKeys, right.columns)
        if args.ids: onKeys = np.intersect1d(onKeys, args.ids)
        log.info(f"Matching on {','.join(onKeys)}")
        # merge where the annotation file annotates, i.e. left join with added info from right file.
        infile = infile.merge(right, on=list(onKeys), how="left", suffixes=('', ' new'))
        # Multiple annotation files can in combination describe the entries in infile, so we replace nans in columns named the same.
        for colname in infile.columns:
            if colname.endswith(" new"):
                colname_old = colname.removesuffix(" new")
                log.debug(f"Annotation replacement on {colname_old}")
                old = infile[colname_old]
                new = infile[colname]
                rowIdx = ~new.isna()
                modIdx = ~(old[rowIdx].isna() | (old[rowIdx] == new[rowIdx]))
                if np.any(modIdx):
                    log.warning(str(modIdx.sum()) + f" modifications on '{colname_old}'.")
                old.update(new)
                infile.drop(columns=colname, inplace=True)
        log.info(f"shape = {infile.shape}")
        
    if infile.isna().any().any():
        # not using ','.join(...) since it becomes long, e.g. if all seq features are involved.
        log.warning(f"NA found in infile in column(s): {infile.columns[infile.isna().any()]}")
    
    #parameter grid generation
    param_grid = param_grid_gen(args.n_estimators, args.criterion, args.max_depth, args.min_samples_split, args.min_samples_leaf, args.max_features)
        
    # do the feature prep and training
    best_param, cv_results_df = run(infile, args.Class, param_grid, args.cv_search, args.metric,random_state=random_state)
    
    # write best parameters
    with open(args.out_param+'.txt','w') as params:
        #print(best_param)
        #print(str(best_param))
        params.write(str(best_param))
    #print(np.array(best_param.keys()))
    #with open(args.out_param+'.txt','w') as csvfile:
        #writer = csv.DictWriter(csvfile, fieldnames=best_param.keys())
        #writer.writeheader()
        #for data in best_param:
            #writer.writerow(data)
    
    # save cv results
    cv_results_df.sort_values(args.metric,ascending=False).to_csv(args.cv_results+'.tsv')
    
if __name__ == '__main__':
    args = get_parser().parse_args()
    args.infile = sys.stdin
    main(args)