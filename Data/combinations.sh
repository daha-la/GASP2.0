#!/usr/bin/env zsh

# Script for creating combination datasets for the GASP pipeline.
# A combination dataset is a dataset that contains all combinations of proteins and acceptors.
# The proteins are all proteins in the input fasta file, and the acceptors are either all acceptors in the acceptor file, or a subset of the acceptors in the acceptor file.
# The script takes two arguments:
# - The name of the fasta file (without the .faa extension)
# - A potential list of target acceptors to use. If this is not provided, all acceptors in the acceptor file will be used.
#   The acceptors should be in the format of "acceptor1 acceptor2 acceptor3" etc.
# The script wil save the combination dataset as a tsv file with tab delimiter in the combinations folder, potentially with a suffix if target acceptors are provided.

# Input variables
name=$1
target_acceptors=$2

# Acceptors file
acceptor="../Representation_Strategies/substrate/encodings/acceptors3_features.tsv"

# Input and output files
infile="fasta/$name.faa"
outfile="combinations/$name"

python ../src/combinations.py -in $infile -out $outfile -acc "$acceptor" -targets $target_acceptors
