#!/usr/bin/env python3
import argparse
from degnutil import argument_parsing as argue, bio_util as bio
from Bio import SeqIO
from Bio.File import as_handle


def get_parser():
    parser = argparse.ArgumentParser(description="Write unwrapped sequences.")
    argue.inoutfile(parser)
    return parser

def parse_args(args):
    return argue.parse_inoutfile(args)

def main(args):
    args = parse_args(args)
    with as_handle(args.outfile, 'w') as outfile:
        for record in SeqIO.parse(args.infile, "fasta"):
            bio.write(str(record.seq), record.description, outfile)


if __name__ == '__main__':
    main(get_parser().parse_args())
