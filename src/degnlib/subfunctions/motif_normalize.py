#!/usr/bin/env python3
import argparse
import numpy as np
from degnutil import argument_parsing as argue, motif_util as mot

def get_parser():
    parser = argparse.ArgumentParser(description="Scale values in arrays so rows sum to 1.")
    argue.inoutfile(parser, help="MEME minimal format.")
    parser.add_argument("-n", "--nan", action="store_true", help="Avoid NaNs by setting columns with all zeros to all ones before normalizing.")
    
    return parser


def parse_args(args):
    return argue.parse_inoutfile(args)


def normalize(arr):
    """
    Scale rows so they each sum to 1.
    :param arr: numpy or pandas array
    :return: numpy or pandas array
    """
    return arr / np.sum(arr, axis=1)[:, np.newaxis]


def main(args):
    args = parse_args(args)
    
    memes = mot.read_meme(args.infile)
    
    for name, motif in memes["matrices"].items():
        if args.nan:
            try: motif.iloc[np.where(np.all(motif == 0, axis=1))[0], :] = 1
            except AttributeError: motif[np.all(motif == 0, axis=1), :] = 1
        memes["matrices"][name] = normalize(motif)
    
    mot.write_meme(args.outfile, **memes)


if __name__ == '__main__':
    main(get_parser().parse_args())

