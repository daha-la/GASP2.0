#!/usr/bin/env python3
import argparse
from argparse import RawTextHelpFormatter
import sys
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging as log
import re


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


# functions

def testing(clf, df):
    testset = df[clf.feature_names_in_]
    # predict on records with NA and write NA as pred for those entries so the output file will not change shape.
    noNaRows = ~testset.isna().any(axis=1)
    if not noNaRows.all(): log.warning(f"NaN records removed for testing: {len(noNaRows)} -> {sum(noNaRows)}")
    predictions = clf.predict_proba(testset[noNaRows])[:, clf.classes_ == 1].squeeze()
    df.loc[noNaRows, "pred"] = predictions
    return df

def main(args):
    if args.log is None: handlers = None
    else: handlers = [log.StreamHandler(), log.FileHandler(args.log)]
    log.basicConfig(format="%(message)s", level=log.INFO, handlers=handlers)
    
    # reading
    
    infile = pd.read_table(args.infile)
    log.info(f"shape = {infile.shape}")
    
    # the original keys, before any potential annotations are added
    infileKeys = infile.columns
    print(infileKeys)
    if args.annotate:
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
        log.warning(f"NA found in infile in column(s): {infile.columns[infile.isna().any()]}")

    
    # testing
    log.info("Read trained model.")
    clf = joblib.load(args.model)
    clfs = [clf] # in case of writing feature importance below.
    log.info("Make prediction on the test set.")      
    test = testing(clf, infile)
    
    # write columns out that were given in infile, i.e. if annotations were made, they are discarded here.
    test[list(infileKeys) + ["pred"]].to_csv(sys.stdout, sep="\t", index=False)
    
    if args.importance is not None:
        importance = np.vstack([clf.feature_importances_ for clf in clfs]).mean(axis=0)
        stds = np.std([t.feature_importances_ for clf in clfs for t in clf.estimators_], axis=0)
        importance_table = pd.DataFrame({"feature": clf.feature_names_in_, "importance": importance, "std": stds})
        importance_table.to_csv(args.importance, sep='\t', index=False)

if __name__ == '__main__':
    args = get_parser().parse_args()
    args.infile = sys.stdin
    main(args)

