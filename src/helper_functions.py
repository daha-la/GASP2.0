import numpy as np
import pandas as pd
from typing import Union
import logging as log
import Bio.PDB as bpdb
from Bio import AlignIO
assert log
import os
import subprocess
from itertools import product

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

def remove_redundant(*dfs: Union[pd.DataFrame, list], ignore: Union[None, set] = None):
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

def feature_prep(train: pd.DataFrame, ignore: Union[None, set] = None):
    """
    Prepare features for model training. This means we make sure to drop string features and redundant features that never vary.

    Args:
        train (pd dataframe): training data
        ignore (Union[None, set]): columns to ignore that are not features

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

def hmmbuilding(input_msa: str, hmm_name: str,hmm_model_path: str = '/Users/dahala/GitHub/GASP2.0/Representation_Strategies/protein/methods/Alignment/hmm_models/',hmmer_path: str = '/Users/dahala/Projects/HMMER/bin'):
    """
    Build an HMM model from a multiple sequence alignment using HMMER.
    Output is saved as a .hmm file in the specified directory.

    Args:
        input_msa (str): Path to the input multiple sequence alignment file.
        hmm_name (str): Name of the HMM model.
        hmm_model_path (str): Path to the directory where the HMM model will be saved.
        hmmer_path (str): Path to the HMMER installation.
    """

    # Define the path to the HMM model
    model_path = hmm_model_path+hmm_name

    # Check if the model already exists, and build it if not
    if not os.path.exists(model_path):
        print('Building HMM model...')
        subprocess.run(f'{hmmer_path}/hmmbuild --amino {model_path}.hmm {input_msa} > {model_path}_hmmbuild.log', shell=True, executable="/bin/zsh")
    
def hmmaligning(input_fastas: list, hmm_name: str, output_name: str,hmm_model_path: str = '/Users/dahala/GitHub/GASP2.0/Representation_Strategies/protein/methods/Alignment/hmm_models/',
                hmmer_path: str = '/Users/dahala/Projects/HMMER/bin',threshold: float = 1):
    """
    Align multiple sequences to an HMM model using HMMER.
    Output is saved as a .afa file in the specified directory.

    Args:
        input_fastas (list): List of paths to the input FASTA files.
        hmm_name (str): Name of the HMM model.
        output_name (str): Name of the output alignment file.
        hmm_model_path (str): Path to the directory where the HMM model is saved.
        hmmer_path (str): Path to the HMMER installation.
        threshold (float): Maximum gap percentage allowed in the alignment.
    """

    # Define the path to the HMM model and the output file
    model_path = hmm_model_path+hmm_name+'.hmm'
    outpath = f'/Users/dahala/GitHub/GASP2.0/Representation_Strategies/protein/methods/Alignment/alignments/{output_name}'
    
    # Make a single string of all input FASTA paths to pass to the HMMER command
    input_fasta_paths = " ".join(input_fastas)

    # Run the HMMER command to align the sequences to the model
    subprocess.run(f'cat {input_fasta_paths} | {hmmer_path}/hmmalign --trim --amino --outformat A2M {model_path} - > {outpath}_raw.A2M', shell=True, executable="/bin/zsh")
    
    # Clean the A2M file and save the cleaned alignment as an AFA file
    clean_a2m_file(f'{outpath}_raw.A2M', f'{outpath}.afa')

    # Read the cleaned alignment and save only sequences with a gap percentage below the threshold
    hmm_msa = AlignIO.read(f'{outpath}.afa', 'fasta')
    skip_count = 0
    with open(f'{outpath}_{hmm_name}.afa','w') as out_file:
        for prot in hmm_msa:
            if gaplimiter(str(prot.seq),threshold=threshold):
                out_file.write('>' + prot.id + '\n')
                out_file.write(str(prot.seq))
                out_file.write('\n')
            else:
                print(f'{prot.id} has too many gaps, skipping...')
                skip_count += 1
    print(f'Skipped {skip_count} sequences due to too many gaps')

def clean_a2m_file(input_file: str, output_file: str):
    """
    Cleans an A2M alignment file:
    - Removes non-consensus residues (lowercase letters)
    - Removes '.' characters
    - Keeps headers intact
    Writes the cleaned alignment to a new file.
    
    Args:
        input_file (str): Path to the input A2M file.
        output_file (str): Path to the cleaned output file.
    """

    # Define a function to remove lowercase residues and '.' characters from a sequence
    def fix_seqs(input_string):
        """Remove lowercase residues and '.' from a sequence."""
        return ''.join([char for char in input_string if char.isupper() or char == '-'])
    
    # Open the input and output files and clean the sequences
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:

        # Iterate over the lines in the input file
        for line in infile:

            # Check if the line is a header, as these should be kept intact
            if line.startswith('>'):  # Header line
                outfile.write(line)
            else:  # Sequence line
                cleaned_seq = fix_seqs(line.strip())
                outfile.write(cleaned_seq + '\n')

def gaplimiter(seq: str,threshold: int = 1):
    """
    Check if the gap ratio of an aligned sequence is below a specified threshold.
    If the gap ratio is above the threshold, return False. Otherwise, return True.

    Args:
        seq (str): An aligned protein sequence.
        threshold (int): Maximum gap ratio allowed in the sequence. Default is 1, meaning all sequences are allowed.

    Returns:
        bool: False if the gap ratio is above the threshold, True otherwise.
    """

    # Calculate the gap ratio of the sequence
    gap_ratio = seq.count('-') / len(seq)

    # Check if the gap ratio is below the threshold, and return True if it is
    if gap_ratio > threshold:
        return False
    else:
        return True
    
def align_to_df(align):
    """
    This function takes a multiple sequence alignment and converts it to a pandas dataframe.

    Args:
        align: A multiple sequence alignment object from AlignIO.read()

    Returns:
        df (pd.DataFrame): A pandas dataframe with the alignment
    """
    
    # Initialize the seq and ID list
    ID_list = []
    seq_list = []

    # Iterate over the alignment and append the sequences and IDs to the lists
    for prot in align:
        ID_list.append(prot.id)
        seq_list.append(prot.seq)

    # Create a pandas dataframe
    df = pd.DataFrame(seq_list, index=ID_list, columns=np.arange(1, len(align[0].seq) + 1))

    return df
    
def encoding_alignment(align,encoding: pd.DataFrame,remove_Cterm=False):
    '''
    Encode all proteins in an alignment using the given sequence encoding

    Args:
        align: A multiple sequence alignment object from AlignIO.read()
        encoding (pd.DataFrame): Sequence encoding

    Returns:
        encoded (pd.DataFrame): DataFrame of encoded proteins
    '''

    # Convert alignment to pandas dataframe
    df = align_to_df(align)

    # Remove C-terminal half of proteins if specified
    if remove_Cterm:
        df = df.iloc[:,:int(df.shape[1]/2)]

    # Make column list to name features by combining name of df columns (alignment positions) and sequence encoding identifiers
    cols = [str(pos)+' '+str(prop) for pos, prop in product(df.columns.to_list(),encoding.index)]

    # Encode all proteins using the given sequence encoding
    encoded = []
    for i in range(len(df.index)):
        encoded.append(encoding[df.iloc[i].to_numpy()].to_numpy().flatten())
    encoded = pd.DataFrame(encoded,index=df.index,columns=cols)

    # Rename the index for use in modular architecture
    encoded.index.names = ['enzyme']

    return encoded

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