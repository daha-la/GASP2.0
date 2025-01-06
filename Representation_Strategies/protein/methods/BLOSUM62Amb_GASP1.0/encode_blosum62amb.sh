#!/usr/bin/env zsh

# This script is used to encode new sequences using the blosum62Amb encoding.
# The INFILE is a fasta file with the original GASP sequences, 'alignment/seqs.faa'. The NEWSEQ is a fasta file with the new sequences to be encoded.
# Both files are concatenated and aligned to the HMM. The sequences are then encoded using the blosum62Amb encoding.
# THRESH is the alignment quality threshold for removing low quality alignments. Lower number to accept more sequences. Set to -1 to use all sequences.
# Remember to have installed muscle. Can be done with conda install muscle -c bioconda

# Define the path to subfunctions and degnlib
LIB=../../../../src/degnlib/subfunctions
export PYTHONPATH="$PYTHONPATH:../../../../src/degnlib/"

# Name of the file with the new sequences to be encoded. Should be in ../data/fasta/ folder using the .faa format.
# Given as an argument to the script.
new_seqs=$1

# Define the input files
INFILE=`ls ../alignment/seqs.faa`

# The alignment quality threshold for removing low quality alignments. Lower number to accept more sequences, -1 to use all sequences.
THRESH=-1

# Align all sequences (both old and new) to the HMM
cat $INFILE | bash hmmalign.sh $THRESH
gzip -c muscle_qual.hmm.nterm.tsv > muscle_qual.hmm.nterm.tsv.gz

# Check which sequences were filtered out due to low quality alignment
mlr --hi join --np --ul -j 1 -f <(grep '>' muscle.hmm.faa) <(grep '>' muscle_qual.hmm.faa) | sed 1d | cut -c2- > discarded.enz
if [ -s discarded.enz ]; then
    echo "The following enzymes were discarded due to gappy alignment when using $THRESH as threshold value:"
    cat discarded.enz
fi

# Make a table that has both aligned and unaligned sequences.
cat $INFILE | $LIB/fasta_table.py -i enzyme -s seq_unaligned |
    mlr -t join -j enzyme -f muscle_qual.hmm.nterm.tsv.gz |
    gzip -c > muscle_qual_wUnalign.hmm.nterm.tsv.gz

# Encode all sequences with the blosum62amb encoding (BLOSUM62 with ambiguous codes).
# All features are kept, but one could use -k flag to ensure.
gunzip -c muscle_qual.hmm.nterm.tsv.gz |
    python ../../../../src/encode_features.py -i enzyme --aa seq --aa-encoding ../../encodings/raw/blosum62Amb.tsv |
    gzip -c > ../../encodings/blosum62Amb_gasp1.tsv.gz