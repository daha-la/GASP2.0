#!/usr/bin/env python3

"""
Conducts a GridSearch on a Random Forest classifier.
Read a tab-separated table with header as well as a list of model parameters from the command line.
Writes the best parameters to a text file and cross-validation results to a tsv file.
Infile columns:
    (-c) A class column where 1 is for a positive and 0 for a negative observation.
    (-F) Feature columns, either here on using -a/--annotate.
    (-i) Optionally id column(s) that can map to features from (-a).
"""

# Importing the required libraries
import argparse
from argparse import RawTextHelpFormatter
from typing import Union
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging as log
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
    average_precision_score,
    make_scorer
)
from src.helper_functions import hit_rate, param_into_list, feature_prep

# Dictionary for evaluation metrics
metric_dict = {"precision": make_scorer(precision_score),
               "recall": make_scorer(recall_score),
                "accuracy": make_scorer(accuracy_score),
                "balanced-accuracy": make_scorer(balanced_accuracy_score),
                "mcc": make_scorer(matthews_corrcoef),
                "f1": make_scorer(f1_score),
                "auc-roc": make_scorer(roc_auc_score),
                "ndcg": make_scorer(ndcg_score),
                "ndcg_top10": make_scorer(ndcg_score,k=10),
                "average_precision": make_scorer(average_precision_score),
                "hit_rate_top1": make_scorer(hit_rate, k=1),
                "hit_rate_top10": make_scorer(hit_rate, k=10)
                }

# Set up the argument parser
def get_parser():
    parser = argparse.ArgumentParser(description="""
    Conduct a GridSearch on a Random Forest classifier. 
    Read a tab-separated table with header as well as a list of model parameters from the command line.
    Writes the best parameters to a text file and cross-validation results to a tsv file.
    Infile columns:
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
    parser.add_argument("-c", "--class", dest="Class",# uppercase Class since class is a reserved word
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
    
def param_grid_gen(n_estimators: Union[int,list], criterion: Union[str,list], max_depth: Union[int,list], min_samples_split: Union[int,list], min_samples_leaf: Union[int,list], max_features: Union[str,int,list]):
    """
    Construct parameter grid for GridSearch

    Args:
        n_estimators (Union[int,list]): number of trees
        criterion (Union[str,list]): criterion for split quality
        max_depth (Union[int,list]): maximum depth of trees
        min_samples_split (Union[int,list]): minimum sample required for split
        min_samples_leaf (Union[int,list]): minimum sample in leaf
        max_features (Union[int,list]): number of features considered when looking for best split
    Return:
        dict: dict of parameter options
    """

    # Ensure that all parameters are lists
    n_estimators = param_into_list(n_estimators)
    criterion = param_into_list(criterion)
    max_depth = param_into_list(max_depth)
    min_samples_split = param_into_list(min_samples_split)
    min_samples_leaf = param_into_list(min_samples_leaf)
    max_features = param_into_list(max_features)

    # Append None to max_depth if not already present
    if max_depth != [None]:
        max_depth.append(None)
    
    # Construct parameter grid
    param_grid = {"n_estimators": n_estimators,
                  "criterion": criterion,
                  "max_features": max_features,
                  "max_depth": max_depth,
                  "min_samples_split": min_samples_split,
                  "min_samples_leaf": min_samples_leaf}
    return param_grid

def grid_search(train: pd.DataFrame, Class: str, param_grid: dict, cv_search: int, metric: str, random_state: int):

    """
    Conducting GridSearch on Random Forest Classifier using given parameter grid

    Args:
        train (pd.DataFrame): training data
        Class (str): name of Class column in training data
        param_grid (dict): dict of parameter options
        cv_search (int): k for crossvalidation
        metric (str): evaluation metric for search
        random_state (int): random state for RandomForestClassifier

    Returns:
        best_params: dict of best parameter options for best model
        cv_results_df: dataframe of cross-validation results
    """

    # Get evaluation metric
    eval_metric = metric_dict[metric]

    # Conduct GridSearch
    estimator = RandomForestClassifier(random_state=random_state)
    gridsearch = GridSearchCV(estimator = estimator, param_grid = param_grid, cv = cv_search, verbose=0, n_jobs = -1, scoring=eval_metric)
    
    # Get best parameters and cross-validation results
    gridsearch.fit(train.drop(columns=Class), train[Class])
    best_params = gridsearch.best_params_
    cv_results = gridsearch.cv_results_

    # Construct dataframe of cross-validation results
    cv_results_df = pd.concat([pd.DataFrame(cv_results["params"]),pd.DataFrame(cv_results["mean_test_score"], columns=[metric]),pd.DataFrame(cv_results["std_test_score"], columns=[metric+'_std'])],axis=1)
    
    return best_params, cv_results_df


def run(train: pd.DataFrame, Class: str, param_grid: dict, cv_search: int, metric: str, random_state: int, ignore: Union[None, list] = None):
    """
    Function for running the GridSearch on the training data. Returns the best parameters and cross-validation results.

    Args:
        train (pd.DataFrame): training data
        Class (str): name of Class column in training data
        param_grid (dict): dict of parameter options
        cv_search (int): k for crossvalidation
        metric (str): evaluation metric for search
        random_state (int): random state for RandomForestClassifier
        ignore (Union[None, list]): columns to ignore that are not features

    Returns:
        best_params: dict of best parameter options for best model
        cv_results_df: dataframe of cross-validation results
    """

    # Prepare features
    if ignore is None: ignore = set() # empty set
    elif np.isscalar(ignore): ignore = {ignore} # single element
    else: ignore = set(ignore) # full set

    # Drop columns that are not features, but keep the class column
    ignore = ignore - {Class}
    train = train.drop(columns=ignore, errors='ignore')
    train = feature_prep(train, ignore.union({Class}))

    # Remove NaN records
    if train.isna().any().any():
        nBefore = len(train)
        train = train.dropna()
        log.warning(f"NaN records removed for training: {nBefore} -> {len(train)}")
    
    # Conduct GridSearch
    log.info("Conduct GridSearch.")
    best_params, cv_results_df = grid_search(train, Class, param_grid, cv_search, metric, random_state)
    
    return best_params, cv_results_df


def main(args):
    """
    Main function for running the GridSearch on the training data. 
    Saves the best parameters and cross-validation results to files.

    Args:
        args: arguments from the parser
    """

    # Logging setup
    if args.log is None: handlers = None
    else: handlers = [log.StreamHandler(), log.FileHandler(args.log)]
    log.basicConfig(format="%(message)s", level=log.INFO, handlers=handlers)
        
    # Read infile containing training data    
    infile = pd.read_table(args.infile)
    log.info(f"shape = {infile.shape}")
    
    # Set random state
    if args.random_state == None:
        random_state = None
    else:
        random_state = int(args.random_state)
    
    # Save the original keys, before any potential annotations are added
    infileKeys = infile.columns

    # Iterate over annotation files
    for infname in args.annotate:
        log.info(f"Annotating from {infname}")

        # Read annotation file
        right = pd.read_table(infname)
        
        # Merge on keys overlapping between the the original keys and annotation files. Use those that are given as ids, if ids are given.
        onKeys = np.intersect1d(infileKeys, right.columns)
        if args.ids: onKeys = np.intersect1d(onKeys, args.ids)
        log.info(f"Matching on {','.join(onKeys)}")

        # Merge training data file with the annotations, i.e. left join with added info from right file.
        infile = infile.merge(right, on=list(onKeys), how="left", suffixes=('', ' new'))
        
        # Multiple annotation files can in combination describe the entries in infile, so we replace nans in columns named the same.
        for colname in infile.columns:

            # Identify new version of old columns
            if colname.endswith(" new"):
                colname_old = colname.removesuffix(" new")
                log.debug(f"Annotation replacement on {colname_old}")
                old = infile[colname_old] # old is the original column
                new = infile[colname] # new is the annotation column
                
                # Extract non-nan rows in new, find modifications, and update old
                rowIdx = ~new.isna()
                modIdx = ~(old[rowIdx].isna() | (old[rowIdx] == new[rowIdx]))
                if np.any(modIdx):
                    log.warning(str(modIdx.sum()) + f" modifications on '{colname_old}'.")
                old.update(new)

                # Remove the new column
                infile.drop(columns=colname, inplace=True)
        log.info(f"shape = {infile.shape}")
    
    # Check for NA in infile, meaning that there are issues with the encoding or the data.
    if infile.isna().any().any():
        # Not using ','.join(...) since it becomes long, e.g. if all seq features are involved.
        log.warning(f"NA found in infile in column(s): {infile.columns[infile.isna().any()]}")
    
    # Parameter grid generation
    param_grid = param_grid_gen(args.n_estimators, args.criterion, args.max_depth, args.min_samples_split, args.min_samples_leaf, args.max_features)
        
    # Run GridSearch
    best_param, cv_results_df = run(infile, args.Class, param_grid, args.cv_search, args.metric,random_state=random_state)
    
    # Write best parameters to txt file
    with open(args.out_param+'.txt','w') as params:
        params.write(str(best_param))
    
    # Save cv results to tsv file
    cv_results_df.sort_values(args.metric,ascending=False).to_csv(args.cv_results+'.tsv')
    
if __name__ == '__main__':
    args = get_parser().parse_args()
    args.infile = sys.stdin
    main(args)