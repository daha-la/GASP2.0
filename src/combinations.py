#!/usr/bin/env python3
"""
This script creates a combinatory file of substrates and proteins in the format used by the GASP models.
The input is a fasta file, an acceptor file, and potentially target acceptors.
If no targets are specified, all possible combinations from the substrate file and fasta file are created.

The output is a tsv file with all possible combinations of substrates and proteins, consisting of two columns: enzyme and substrate specifier (currently only CID supported).

"""

# Importing libraries
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO

# Defining the parser
def get_parser():
    parser = argparse.ArgumentParser(description="Create combinatory file of all substrates and every protein in fasta file.")
    parser.add_argument("-in", "--infile", help='Path of the fasta file')
    parser.add_argument("-acc", "--acceptor_file", help='Path of acceptor file')
    parser.add_argument("-out", "--outfile", help='Path of out file')
    parser.add_argument("-targets","--target_acceptors", nargs='+', default=None,help='Target acceptors') # CID of target acceptors
    return parser

# Main function
def main(args):

    # If target acceptor is specified, only combinations with this target acceptor are created
    if args.target_acceptors is not None:
        
        # Extracting the target acceptors from the arguments
        cids = list(args.target_acceptors)
        cids = [int(i) for i in cids]
        cids = np.array(cids)

    # If no target acceptor is specified, all possible combinations from the substrate file are created
    else:
        substrate_file = pd.read_csv(args.substrates,delimiter='\t')
        cids = np.unique(substrate_file.cid)

    # Extracting protein names from fasta file
    protein_names = []
    for record in SeqIO.parse(args.infile, "fasta"):
        protein_names.append(record.description)

    # Creating a meshgrid of all possible combinations
    mesh = np.array(np.meshgrid(protein_names,cids)).T.reshape(-1,2)

    # Saving the meshgrid to a Pandas dataframe and writing it to a tsv file
    df = pd.DataFrame(mesh,columns=['enzyme','cid'])
    if args.target_acceptors is not None:

        # Join cids into a single string
        suffix = '_'.join(f'{i}' for i in cids)
        df.to_csv(args.outfile+'_'+suffix+'.tsv',sep='\t',index=False)
    else:
        df.to_csv(args.outfile+'_all.tsv',sep='\t',index=False)
    
# Running the main function
if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
