#!/usr/bin/env python3
import argparse
from Bio import SeqIO
from Bio.File import as_handle
from degnutil import argument_parsing as argue, bio_util as bio
from degnutil.string_util import delete_lowercase


def get_parser():
    parser = argparse.ArgumentParser(description="Delete lowercase letters from fasta.")
    argue.inoutfile(parser)
    return parser
    

def parse_args(args):
    return argue.parse_inoutfile(args)


def main(args):
    args = parse_args(args)
    with as_handle(args.outfile, 'w') as outfile:
        for record in SeqIO.parse(args.infile, "fasta"):
            bio.write(delete_lowercase(record), record.description, outfile)


if __name__ == '__main__':
    main(get_parser().parse_args())

