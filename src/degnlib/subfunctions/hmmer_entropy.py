#!/usr/bin/env python3
import argparse
import numpy as np
from Bio.File import as_handle
# from scipy.stats import entropy
from degnutil import argument_parsing as argue
from degnutil.hmmer_util import parse_hmmer

def get_parser():
    parser = argparse.ArgumentParser(description="Print entropy for each file. base=2.")
    argue.infiles(parser)
    argue.outfile(parser)
    return parser

def entropy2(x):
    return -sum(x * np.log2(x))

def main(args):
    with as_handle(args.outfile, 'w') as outfile:
        for file in args.infiles:
            match_emissions = np.concatenate(parse_hmmer(file)["match_emission"])
            # outfile.write(str(entropy(match_emissions, base=2)) + '\n')
            outfile.write(str(entropy2(match_emissions)) + '\n')

if __name__ == '__main__':
    parsed_args = get_parser().parse_args()
    main(parsed_args)

