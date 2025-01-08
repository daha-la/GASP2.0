#!/usr/bin/env zsh
name=$1
acceptor="../encodings/acceptors3_features.tsv"
target_acceptor=$2
infile="../data/fasta/$name.faa"
outfile="combinations/$name"

python ../src/combinations.py -in $infile -out $outfile -sub $acceptor -target $target_acceptor