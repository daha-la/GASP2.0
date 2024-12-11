#!/usr/bin/env python3
import argparse
import numpy as np
from Bio.File import as_handle
from degnutil import argument_parsing as argue, bio_util as bio


def get_parser():
    parser = argparse.ArgumentParser(
        description="Delete insertions in the consensus sequence of an alignment (selex from hmmalign).")
    argue.inoutfile(parser)
    return parser
    

def parse_args(args):
    return argue.parse_inoutfile(args)


def main(args):
    args = parse_args(args)
    seqs, names = bio.read_selex(args.infile)
    
    # first record should be consensus indicator
    idx = ~bio.insertion_index(seqs[0])
    
    with as_handle(args.outfile, 'w') as outfile:
        for seq, name in zip(seqs[1:], names[1:]):
            bio.write(''.join(np.char.asarray(list(seq))[idx]), name, outfile)
    

if __name__ == '__main__':
    main(get_parser().parse_args())

