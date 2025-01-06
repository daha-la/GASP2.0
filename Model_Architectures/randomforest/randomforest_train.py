#!/usr/bin/env python3

"""
Trains a Random Forest classifier.
Reads a tab-separated table with header from stdin and saves the model to a file.
Infile columns:
    (-c) A class column where 1 is for a positive and 0 for a negative observation.
    (-F) Feature columns, either here on using -a/--annotate.
    (-i) Optionally id column(s) that can map to features from (-a).
"""

# Importing the required libraries
import argparse
from argparse import RawTextHelpFormatter
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging as log
from typing import Union
import os

# Local imports
current_dir = os.path.abspath('')
project_root = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(project_root)
from src.helper_functions import feature_prep

# Set up the argument parser
def get_parser():
    parser = argparse.ArgumentParser(description="""
    Trains a Random Forest classifier. 
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
    parser.add_argument("-o", "--out",
        help="The model can be saved to this file (e.g. randomforest.joblib.gz).\nBy default the model is not saved.")
    parser.add_argument("-n", "--estimators", dest="n_estimators", type=int, default=100,
        help="Number of trees in the forest.")
    parser.add_argument("-crit", dest="criterion", type=str, default="gini",
        help="Randomforest criterion for split quality.", choices=["gini", "entropy"])
    parser.add_argument("-maxD", "--max_depth", dest="max_depth", default=None,
        help="The maximum depth of the tree.")
    parser.add_argument("-minS", "--min_split", dest="min_samples_split", type=int, default=2,
        help="The minimum number of samples required to split an internal node.")
    parser.add_argument("-minL", "--min_leaf", dest="min_samples_leaf", type=int, default=1,
        help="The minimum number of samples required to be at a leaf node.")
    parser.add_argument("-maxF", "--max_feat", dest="max_features", default="sqrt",
        help="The number of features to consider when looking for the best split.")
    parser.add_argument("--class-weight",
        help="Randomforest class weight. Default=equal weight to each datapoint.", choices=["balanced", "balanced_subsample"])
    parser.add_argument("-c", "--class", dest="Class", # uppercase Class since class is a reserved word
        help="Name of column with class labels in training data. Default=class.", default="class")
    parser.add_argument("--log",
        help="Logging is sent to stderr. Set flag to a filename to also write to file.")
    parser.add_argument("-rand", "--random_state", dest="random_state",
        help="Defines the random state of the RandomForestClassifier", default=42)
    
    return parser

def training(train: pd.DataFrame, Class: str, n_estimators: int, criterion: str, max_depth: int, min_samples_split: int,
              min_samples_leaf: int, max_features: Union[str,int], class_weight: str, random_state: int):
    """
    Perform training of random forest classifier with the given features. Returns the trained model.

    Args:
        train (pd.DataFrame): training dataset with all features annotated.
        Class (str): name of column with class label.
        n_estimators (int): number of trees in the forest.
        criterion (str): criterion for split quality.
        max_depth (int): maximum depth of the tree.
        min_samples_split (int): minimum number of samples required to split an internal node.
        min_samples_leaf (int): minimum number of samples required to be at a leaf node.
        max_features (int): number of features to consider when looking for the best split.
        class_weight (str): class weight.
        random_state (int): random state for reproducibility.
        
    Returns:
        RandomForestClassifier: trained model
    """

    # Train the model
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf, max_features=max_features,class_weight=class_weight,random_state=random_state)
    clf.fit(train.drop(columns=Class), train[Class])
    return clf

def train_func(train: pd.DataFrame, Class: str, n_estimators: int, criterion: str, max_depth: int, min_samples_split: int,
              min_samples_leaf: int, max_features: Union[str,int], class_weight: str, random_state: int, ignore: Union[None, list] = None):
    """
    Function for preparing the training data and training the Random Forest model.
    Returns the trained model.

    Args:
        train (pd.DataFrame): training dataset with all features annotated.
        Class (str): name of column with class label.
        n_estimators (int): number of trees in the forest.
        criterion (str): criterion for split quality.
        max_depth (int): maximum depth of the tree.
        min_samples_split (int): minimum number of samples required to split an internal node.
        min_samples_leaf (int): minimum number of samples required to be at a leaf node.
        max_features (int): number of features to consider when looking for the best split.
        class_weight (str): class weight.
        random_state (int): random state for reproducibility.
        ignore (Union[None, list]): columns to ignore that are not features
    """


    # Prepare features
    if ignore is None: ignore = set() # empty set
    elif np.isscalar(ignore): ignore = {ignore} # single element
    else: ignore = set(ignore) # full set

    # Drop columns that are not features, but keep the class column
    ignore = ignore - {Class}
    print(ignore)
    print(ignore.union({Class}))
    train = train.drop(columns=ignore, errors='ignore')
    train = feature_prep(train, ignore.union({Class}))

    # Remove NaN records
    if train.isna().any().any():
        nBefore = len(train)
        train = train.dropna()
        log.warning(f"NaN records removed for training: {nBefore} -> {len(train)}")

    # Train the model
    log.info("Train random forest model.")
    clf = training(train, Class, n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, class_weight, random_state)
    return clf

def main(args):
    """
    Main function for training a Random Forest classifier. 
    Saves the model to a file specified by the --out argument.
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

    #Fix None parameters
    if args.max_depth == 'None': args.max_depth = None
    
    # Train the model
    clf = train_func(infile, args.Class, args.n_estimators, args.criterion, args.max_depth, args.min_samples_split,
                     args.min_samples_leaf, args.max_features, args.class_weight, random_state, ignore=None)

    # Output the trained model
    joblib.dump(clf, args.out)
    
if __name__ == '__main__':
    args = get_parser().parse_args()
    args.infile = sys.stdin
    main(args)

