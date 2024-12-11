#!/usr/bin/env python3
import argparse
from Bio import SeqIO
from Bio.File import as_handle
from degnutil import argument_parsing as argue, bio_util as bio
import swalign

def get_parser():
    parser = argparse.ArgumentParser(description="Align with Smith-Waterman.")
    argue.inoutfile(parser)
    parser.add_argument("-s", "--seqs", help="Align all seqs in infile to all in this file. "
                                             "Omit to align all-vs-all inside infile.")
    parser.add_argument("--scores", action="store_true", help="Get alignment scores rather than the alignment. Only this is implemented.")

    return parser
    

def parse_args(args):
    args = argue.parse_inoutfile(args)
    if args.seqs is None: args.seqs = args.infile
    if not args.scores: raise NotImplementedError
    args.sep = '\t'
    return args


def main(args):
    args = parse_args(args)
    seqs = list(SeqIO.parse(args.seqs, "fasta"))
    sw = swalign.LocalAlignment(swalign.IdentityScoringMatrix())

    with as_handle(args.outfile, 'w') as outfile:
        outfile.write(args.sep.join(["id"] + [s.description for s in seqs]) + '\n')
        for record in SeqIO.parse(args.infile, "fasta"):
            scores = (sw.align(record.seq, other_record.seq).score for other_record in seqs)
            outfile.write(record.description + args.sep + args.sep.join(map(str, scores)) + '\n')


if __name__ == '__main__':
    main(get_parser().parse_args())

