#!/usr/bin/env python3
import argparse
from degnutil import argument_parsing as argue, motif_util as mot, read_write as rw

def get_parser():
    parser = argparse.ArgumentParser(description="Convert a motif file to MEME format.")
    argue.inoutfile(parser)
    argue.delimiter(parser)
    parser.add_argument("-b", "--bg-freq", nargs="+", default=None,
                        help="Background frequencies for each letter of the alphabet. "
                             "Can be values given directly or as a file.")
    parser.add_argument("-a", "--alphabet",
                        help="Alphabet to use. "
                             "Can also be provided for each PFM as column names.")
    parser.add_argument("-s", "--strand", dest="strands", nargs="+", help="Sequence strands (+ and/or -).")
    parser.add_argument("-n", "--n-samples", 
                        help='Number of samples used creating each of the motifs. Set to "estimate" to estimate the counts.')
    return parser


def parse_args(args):
    args = argue.parse_delimiter(args)
    
    # parse to float vector
    if args.bg_freq is not None:
        if len(args.bg_freq) == 1:
            args.bg_freq = rw.read_vector(args.bg_freq[0], args.delimiter)
        else:
            args.bg_freq = map(float, args.bg_freq)
    
    return argue.parse_inoutfile(args)


def main(args):
    args = parse_args(args)
    
    motifs = mot.read_motifs(args.infile)
    
    if args.n_samples is not None:
        try: args.n_samples = int(args.n_samples)
        except ValueError:
            if args.n_samples != "estimate": raise argparse.ArgumentTypeError("-n/--n-samples value unrecognized")
            args.n_samples = {k:mot.estimate_n_samples(v) for k,v in motifs.items()}
    
    mot.write_meme(args.outfile, motifs, alphabet=args.alphabet, strands=args.strands, bg_freq=args.bg_freq, n_samples=args.n_samples)


if __name__ == '__main__':
    main(get_parser().parse_args())

