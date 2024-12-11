#!/usr/bin/env python3
import argparse
import numpy as np
from degnutil import argument_parsing as argue, motif_util as mot

def get_parser():
    parser = argparse.ArgumentParser(description="Convert PPM to PFM.")
    argue.inoutfile(parser)
    return parser


def parse_args(args):
    return argue.parse_inoutfile(args)


def main(args):
    args = parse_args(args)
    
    memes = mot.read_meme(args.infile)
    
    # get #samples
    if len(memes["n_samples"]) == 0:
        memes["n_samples"] = {k:mot.estimate_n_samples(v) for k,v in memes["matrices"].items()}
    
    memes["matrices"] = {name: (motif*memes["n_samples"][name]).round().astype(int) 
                       for name, motif in memes["matrices"].items()}
    
    mot.write_meme(args.outfile, **memes)


if __name__ == '__main__':
    main(get_parser().parse_args())

