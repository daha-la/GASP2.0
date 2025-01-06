#!/usr/bin/env zsh

# Align sequences to a pre-built HMM, and filter out low quality alignments if necessary.
# Lastly, only the N-terminal half of the alignments are kept.
# WRITES: muscle.hmm.a2m, muscle.hmm.faa, and muscle_qual.hmm.faa, muscle_qual.hmm.nterm.tsv in new/

# The alignment quality threshold for removing low quality alignments. Lower number to accept more sequences
THRESH=$1

# Define the path to subfunctions and degnlib
LIB=../src/degnlib/subfunctions
export PYTHONPATH="$PYTHONPATH:../src/degnlib"

# Align all new sequences to the HMM
hmmalign --trim --amino --outformat A2M ../Representation_Strategies/protein/methods/alignment/GASP.hmm - > new/muscle.hmm.a2m

# Remove non-consensus
$LIB/fasta_delete_lower.py new/muscle.hmm.{a2m,faa}

# Remove low quality alignments
$LIB/fasta_filter_quality.py -t $THRESH new/muscle{,_qual}.hmm.faa

# Use only N-term (discard last half of alignments)
# Get the length of the alignment and divide by 2
length=$($LIB/fasta_length.py new/muscle_qual.hmm.faa | sort -u)
let length=length/2

# Extract the N-terminal half of the alignments
$LIB/fasta_range.py new/muscle_qual.hmm.faa -r 1-$length |
    $LIB/fasta_table.py -i enzyme -s seq > new/muscle_qual.hmm.nterm.tsv