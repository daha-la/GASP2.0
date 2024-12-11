#!/usr/bin/env python3

"""
Test a Random Forest classifier.
Reads a tab-separated table with header from stdin and writes prediction table to stdout.
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
import joblib
import logging as log

# Set up the argument parser
def get_parser():
    parser = argparse.ArgumentParser(description="""
    Test a Random Forest classifier. 
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
    parser.add_argument("--log",
        help="Logging is sent to stderr. Set flag to a filename to also write to file.")
    parser.add_argument("--importance",
        help="Write feature importance and stddev to this file.")
    
    return parser

def testing(clf: object, df: pd.DataFrame) -> pd.DataFrame:
    """
    Test a classifier on a dataframe.

    Args:
        clf: Classifier object.
        df: DataFrame with features and class column.

    Returns:
        DataFrame: DataFrame with predictions.
    """

    # Ensure that the testset has the same features as the training set.
    testset = df[clf.feature_names_in_]

    # Predict on records without NA.
    noNaRows = ~testset.isna().any(axis=1)
    if not noNaRows.all(): log.warning(f"NaN records removed for testing: {len(noNaRows)} -> {sum(noNaRows)}")
    predictions = clf.predict_proba(testset[noNaRows])[:, clf.classes_ == 1].squeeze()

    # Write predictions to the dataframe.
    df.loc[noNaRows, "pred"] = predictions
    return df

def main(args):
    """
    Main function for testing a Random Forest classifier. 
    Saves the predictions as a csv file to stdout.
    If "importance" parameter is given, saves the feature importance to a csv file specified by the parameter.
    Args:
        args: arguments from the parser
    """

    # Logging setup
    if args.log is None: handlers = None
    else: handlers = [log.StreamHandler(), log.FileHandler(args.log)]
    log.basicConfig(format="%(message)s", level=log.INFO, handlers=handlers)
        
    # Read infile containing test data    
    infile = pd.read_table(args.infile)
    log.info(f"shape = {infile.shape}")
    
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

    
    # Read and test the model
    log.info("Read trained model.")
    clf = joblib.load(args.model)
    log.info("Make prediction on the test set.")      
    test = testing(clf, infile)
    
    # Write columns out that were given in infile, i.e. if annotations were made, they are discarded here.
    test[list(infileKeys) + ["pred"]].to_csv(sys.stdout, sep="\t", index=False)
    
    # Write feature importance and respective stds to file if specified
    if args.importance is not None:
        importance = clf.feature_importances_
        stds = np.std([t.feature_importances_ for t in clf.estimators_], axis=0)
        importance_table = pd.DataFrame({"feature": clf.feature_names_in_, "importance": importance, "std": stds})
        importance_table.to_csv(args.importance, sep='\t', index=False)

if __name__ == '__main__':
    args = get_parser().parse_args()
    args.infile = sys.stdin
    main(args)

