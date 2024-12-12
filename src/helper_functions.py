import numpy as np
import pandas as pd
from typing import Union
import logging as log
import Bio.PDB as bpdb
from Bio import AlignIO
assert log                                                                                                                          
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

def alignment_to_embedding(alignment: object, mean_rep_dict: dict):
    """
    Function to convert an alignment to an embedding using the mean representation dictionary.
    Gaps ('-') in the alignment are replaced with zeros in the embedding, while the remaining residues are replaced with their mean representation.
    Requires the mean representation dictionary to be generated beforehand, and in the format {protein_id: mean_representation}.
    The mean representation must have the same length as the protein sequence.

    Args:
        alignment (object): BioPython MSA object.
        mean_rep_dict (dict): Dictionary with the mean representation for each protein in the alignment.

    Returns:
        rep_mean_aligned_df (pd.DataFrame): DataFrame with the mean aligned representation for each protein in the alignment.
        rep_mean_aligned_Nt_df (pd.DataFrame): DataFrame with the mean aligned representation for the N-terminal half of each protein in the alignment.
    """
    # Initialize DataFrame variables to store the mean aligned representations
    rep_mean_aligned_df = None
    rep_mean_aligned_Nt_df = None

    # Loop over the proteins in the alignment
    for prot in alignment:

        # Get the sequence and the protein ID
        seq_ = np.array(prot.seq)
        id_ = prot.id

        # If the dataframe is not fully initialized, do so
        if rep_mean_aligned_df is None:
            rep_mean_aligned_df = pd.DataFrame(columns=np.arange(len(seq_)))
            rep_mean_aligned_Nt_df = pd.DataFrame(columns=np.arange(len(seq_)/2))
        
        # Replace the '/' character in the ID to ensure consistency between the file name and the dictionary key
        if id_.count('/') == 1:
            id_ = id_.replace('/','_')

        # Create an array with the mean representation for the protein, using the alignment to place the values in the correct position
        array_ = np.zeros(len(seq_)) # Initialize the array with zeros, such that the '-' characters will remain as zeros
        array_[np.where(seq_!='-')] = mean_rep_dict[id_]

        # Store the mean aligned representation for the full sequence and the N-terminal half
        rep_mean_aligned_df.loc[id_] = array_
        rep_mean_aligned_Nt_df.loc[id_] = array_[:int(len(array_)/2)]

    return rep_mean_aligned_df, rep_mean_aligned_Nt_df


class NtermSelectPDB(bpdb.Select):
    """
    Class to select residues in the N-terminal half of a protein chain in a PDB file.
    Takes a PDB file and a chain ID as input, and saves the N-terminal half of the chain to a new PDB file.
    """

    def __init__(self, prot_path: str, chain_id: str):
        """
        Initialize the class with the protein structure and chain ID.

        Args:
            prot_path (str): Path to the PDB file.
            chain_id (str): Chain ID of the protein.
        """
        self.chain_id = chain_id

        # Load the protein structure and get the length of the chain
        self.struc = bpdb.PDBParser().get_structure('temp', prot_path)
        self.prot_len = len(self.struc[0][chain_id])  # Length of the chain
        self.start_res = int(self.prot_len / 2)  # Midpoint of the protein
        self.end_res = self.prot_len  # Full length of the protein

    def accept_residue(self, res):
        """
        Exlude residues outside the a specified range in the specified chain.

        Args:
            res: Residue object from Bio.PDB

        Returns:
            bool: True if the residue should be kept, False if it should be excluded.
        """
        # Keep residues outside the second half of the protein
        if self.start_res <= res.id[1] <= self.end_res and res.parent.id == self.chain_id:
            return False  # Exclude these residues
        return True  # Keep all others

    def save_nterm(self, output_path):
        """
        Save the N-terminal half of the protein chain to a new PDB file.

        Args:
            output_path (str): Path to save the new PDB file.
        """

        nterm = bpdb.PDBIO()
        nterm.set_structure(self.struc)
        nterm.save(output_path, self)