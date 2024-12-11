import numpy as np
import pandas as pd
from typing import Union
import logging as log

def param_into_list(param):
    """
    Function to convert parameter into list if it is not already a list

    Args:
        param: parameter to be converted into list
            
    Returns:
        list: list of parameter
    """
    if type(param) != list:
        return [param]
    else:
        return param
    
def hit_rate(true_bin: np.ndarray, predicted_labels: np.ndarray, k: int = 10) -> float:
    """
    Calculate the hit rate based on the top k predicted labels.
    
    Args:
        true_bin (np.ndarray): Binary labels indicating hits (1 for hit, 0 for miss).
        predicted_labels (np.ndarray): Model-predicted scores used to rank the top k entries.
        k (int): The number of top entries to consider.
        
    Returns:
        float: The hit rate as the ratio of hits in the top k entries.
    """

    # Get the top k positions in the predicted_labels array
    top_k_positions = np.argsort(predicted_labels)[-k:]
    
    # Select elements in true_bin based on positions, not index labels
    hits_in_top_k = true_bin.take(top_k_positions)
    
    # Calculate and return hit rate
    return hits_in_top_k.sum() / k

def remove_redundant(*dfs: Union[pd.DataFrame, list], ignore: Union[None, list] = None):
    """
    Find and remove columns where all values are identical across given dataframes.

    Args:
        dfs: pandas dataframes
        ignore: names of columns to ignore

    Returns:
        list: list of strings names for columns that are redundant to use as features
    """
    # skip None dataframes
    dfs = [df for df in dfs if df is not None]

    # Concatenate dataframes and drop columns that are not features
    cat = pd.concat(dfs).drop(columns=ignore)

    # Find columns that are redundant, i.e. never varies
    redundant = cat.columns[np.all(cat == cat.iloc[0, :], axis=0)]
    
    # If redundant columns are found, remove them from the dataframes and print the number of removed columns
    if len(redundant) > 0:
        log.info("Removed {} out of {} ({:.2f}%) features that never varies".
                 format(len(redundant), len(cat.columns), len(redundant) / len(cat.columns) * 100))
        return [df.drop(columns=redundant) for df in dfs]
    
    # If no redundant columns are found, return the original dataframes
    return dfs

def feature_prep(train: pd.DataFrame, ignore: Union[None, list] = None):
    """
    Prepare features for model training. This means we make sure to drop string features and redundant features that never vary.

    Args:
        train (pd dataframe): training data
        ignore (Union[None, list]): columns to ignore that are not features

    Returns:
        modified train set ready for random forest
    """

    # Extract feature names and dtypes
    train_feature_names = np.setdiff1d(train.columns, ignore)
    train_feature_dtypes = train[train_feature_names].dtypes
    # For now drop string features, but in the future they should maybe be implemented as 1-hot automatically
    train_nonum = train_feature_names[(train_feature_dtypes != int) & (train_feature_dtypes != float)]
    if len(train_nonum) > 0:
        log.info("Dropping non-number feature(s): " + ','.join(train_nonum))
        train.drop(columns=train_nonum, inplace=True)

    # Remove redundant features
    train, = remove_redundant(train, ignore=ignore)
    
    return train