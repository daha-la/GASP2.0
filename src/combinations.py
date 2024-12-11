#!/usr/bin/env python3
"""
This script creates a combinatory file of all substrates and every protein in fasta file.
The input is a fasta file, a substrate file, and potentially a target acceptor.
The output is a tsv file with all possible combinations of substrates and proteins.

If no target acceptor is specified, all possible combinations from the substrate file are created.
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
    parser.add_argument("-sub", "--substrates", help='Path of substrate file')
    parser.add_argument("-out", "--outfile", help='Path of out file')
    parser.add_argument("-target","--target_acceptor", nargs='?', default=None,help='Target acceptor') # CID of target acceptor
    return parser

# Main function
def main(args):

    # If target acceptor is specified, only combinations with this target acceptor are created
    if args.target_acceptor is not None:
        cids = int(args.target_acceptor)

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
    if args.target_acceptor is not None:
        df.to_csv(args.outfile+'_'+str(cids)+'.tsv',sep='\t',index=False)
    else:
        df.to_csv(args.outfile+'.tsv',sep='\t',index=False)
    
# Running the main function
if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
