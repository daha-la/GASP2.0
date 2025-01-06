#!/usr/bin/env zsh

# This script is used to obtain an MSA and build a HMM from a set of sequences using muscle and hmmbuild
# The input is a fasta file with the sequences to be aligned
# The output an MSA
# Furthermore, a HMM is built from the aligned sequences

# Remember to install muscle and HMMER before running this script

# Define the path to the muscle executable
muscle_exe="muscle"

# Define the input and output files

IN_FASTA_NAME="seqs"
MODEL_NAME="GASP"

# Align the sequences using muscle
$muscle_exe -align ../../../../Data/fasta/$IN_FASTA_NAME.faa -output alignments/muscle.afa > muscle.log

# Build a HMM from the aligned sequences
hmmbuild --amino alignment/hmm_models/$MODEL_NAME.hmm alignments/muscle.afa > alignment/hmm_logs/$MODEL_NAME.hmm.log

