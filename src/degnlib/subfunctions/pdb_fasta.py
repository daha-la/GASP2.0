#!/usr/bin/env python3
import argparse
from Bio.File import as_handle
from pathlib import Path
from degnutil import argument_parsing as argue, bio_util as bio

def get_parser():
    parser = argparse.ArgumentParser(description="Extract all sequences from a pdb file including start index.")
    argue.infiles(parser)
    argue.outfile(parser)
    return parser

def main(args):
    with as_handle(args.outfile, 'w') as outfile:
        for infile in args.infiles:
            name = Path(infile).stem
            structure = bio.read_pdb(infile)
            for seq, chain, origin, alphabet in bio.yield_pdb_sequences(structure):
                bio.write(seq, "{}:{} origin={} alphabet={}".format(name, chain, origin, alphabet), outfile)


if __name__ == '__main__':
    main(get_parser().parse_args())

