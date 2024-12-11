#!/usr/bin/env zsh

# This script is used to encode new sequences using the blosum62Amb encoding.
# The INFILE is a fasta file with the original GASP sequences, 'alignment/seqs.faa'. The NEWSEQ is a fasta file with the new sequences to be encoded.
# Both files are concatenated and aligned to the HMM. The sequences are then encoded using the blosum62Amb encoding.
# THRESH is the alignment quality threshold for removing low quality alignments. Lower number to accept more sequences. Set to -1 to use all sequences.
# Remember to have installed muscle. Can be done with conda install muscle -c bioconda

# Define the path to subfunctions and degnlib
LIB=../degnlib/subfunctions
export PYTHONPATH="$PYTHONPATH:../degnlib"

# Name of the file with the new sequences to be encoded. Should be in ../data/fasta/ folder using the .faa format.
# Given as an argument to the script.
new_seqs=$1

# Define the input files
INFILE=`ls ../alignment/seqs.faa`
NEWSEQ=../data/fasta/${new_seqs}.faa

# The alignment quality threshold for removing low quality alignments. Lower number to accept more sequences, -1 to use all sequences.
THRESH=-1

# Align all sequences (both old and new) to the HMM
cat $INFILE $NEWSEQ | bash hmmalign_newseq.sh $THRESH
gzip -c new/muscle_qual.hmm.nterm.tsv > new/muscle_qual.hmm.nterm.tsv.gz

# Check which sequences were filtered out due to low quality alignment
mlr --hi join --np --ul -j 1 -f <(grep '>' new/muscle.hmm.faa) <(grep '>' new/muscle_qual.hmm.faa) | sed 1d | cut -c2- > new/discarded.enz
if [ -s new/discarded.enz ]; then
    echo "The following enzymes were discarded due to gappy alignment when using $THRESH as threshold value:"
    cat new/discarded.enz
fi

# Make a table that has both aligned and unaligned sequences.
cat $INFILE $NEWSEQ | $LIB/fasta_table.py -i enzyme -s seq_unaligned |
    mlr -t join -j enzyme -f new/muscle_qual.hmm.nterm.tsv.gz |
    gzip -c > new/muscle_qual_wUnalign.hmm.nterm.tsv.gz

# Encode all sequences with the blosum62amb encoding (BLOSUM62 with ambiguous codes).
# All features are kept, but one could use -k flag to ensure.
gunzip -c new/muscle_qual.hmm.nterm.tsv.gz |
    python encode_features.py -i enzyme --aa seq --aa-encoding ../encodings/raw/blosum62Amb.tsv |
    gzip -c > ../encodings/blosum62Amb_${new_seqs}.tsv.gz