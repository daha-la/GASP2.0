#!/usr/bin/env zsh

# This script is used to obtain an MSA and build a HMM from a set of sequences using muscle and hmmbuild
# The input is a fasta file with the sequences to be aligned
# The output an MSA
# Furthermore, a HMM is built from the aligned sequences

# Remember to install muscle and HMMER before running this script

#Define the folder
folder="/Users/dahala/GitHub/GASP2.0/Representation_Strategies/protein/methods/alignment"

# Define the path to the muscle executable
muscle_exe="$folder/muscle"

# Define the input and output files

IN_FILE="$folder/seqs.faa"
OUT_FILE="$folder/seqs.afa"

# Align the sequences using muscle
$muscle_exe -align $IN_FILE -output $OUT_FILE > muscle.log

# Build a HMM from the aligned sequences
hmmbuild --amino $folder/hmm_models/GASP.hmm $OUT_FILE > $folder/hmm_logs/GASP.hmm.log

