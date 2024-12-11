#!/usr/bin/env python3
import argparse
import numpy as np
from degnutil import argument_parsing as argue, motif_util as mot

def get_parser():
    parser = argparse.ArgumentParser(description="Take mean of MEME file.")
    argue.inoutfile(parser)
    parser.add_argument("-n", "--name", default="mean_motif", help="Name the resulting motif.")
    parser.add_argument("-w", "--weighted", action="store_true", 
                        help="Mean weighted by nsites (#samples) for each motif. Default is equal weight to each motif.")
    
    return parser


def parse_args(args):
    return argue.parse_inoutfile(args)


def main(args):
    args = parse_args(args)
    
    memes = mot.read_meme(args.infile)
    
    # get #samples
    if len(memes["n_samples"]) == 0:
        memes["n_samples"] = {k:mot.estimate_n_samples(v) for k,v in memes["matrices"].items()}
    n_samples = sum(memes["n_samples"].values())
    
    # take average
    if args.weighted:
        empPPM = sum(v * memes["n_samples"][k] for k,v in memes["matrices"].items()) / n_samples
    else:
        empPPM = sum(memes["matrices"].values()) / len(memes["matrices"])
    
    # write
    memes["matrices"] = {args.name: empPPM}
    mot.write_meme(args.outfile, **memes)


if __name__ == '__main__':
    main(get_parser().parse_args())

