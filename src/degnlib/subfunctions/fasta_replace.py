#!/usr/bin/env python3
import argparse
import numpy as np
import re
from Bio import SeqIO
from Bio.File import as_handle
from degnutil import argument_parsing as argue, bio_util as bio, read_write as rw, string_util as st


def get_parser():
    parser = argparse.ArgumentParser(description="Replace codes (characters). Default is random replacement from the alphabet found in the file.")
    argue.inoutfile(parser)
    parser.add_argument("-r", "--replace", help="Only replace these codes (characters).")
    parser.add_argument("-a", "--alphabet", help="Sample from this alphabet. Default is from the codes found in the infile (only letters, not gap characters).")
    parser.add_argument("-f", "--frequencies", type=float, nargs="*", 
        help="Sample from the alphabet with these frequencies. Default is equal proportions. Set flag with nargs=0 to use frequencies found in infile.")
    
    return parser
    

def parse_args(args):

    if args.replace is not None: args.replace = list(args.replace)
    if args.alphabet is not None: args.alphabet = list(args.alphabet)

    return argue.parse_inoutfile(args)


"""
Get all (non-unique) codes from the sequence in each record.
:return: np array of characters
"""
def get_codes(records):
    codes = np.concatenate([list(r.seq) for r in records])
    return np.asarray(re.findall('[A-Za-z]', "".join(codes)))


"""
Change codes in a sequence to a new codes randomly sampled from an alphabet.
"""
def randomize(seq, alphabet, frequencies=None, replace=None):
    out = np.random.choice(alphabet, len(seq), p=frequencies)
    
    if replace is not None:
        seq = np.asarray(list(seq))
        # make index for characters that will NOT be replaced
        idx = np.ones(len(seq), dtype=bool)
        for r in replace:
            idx &= seq != r
        out[idx] = seq[idx]
    
    return "".join(out)


def main(args):
    args = parse_args(args)
    
    records = list(SeqIO.parse(args.infile, "fasta"))
    codes = None

    if args.alphabet is None:
        codes = get_codes(records)
        args.alphabet = np.unique(codes)

    if args.frequencies is None:
        pass # nothing has to be done since None works as equal frequencies in the np.random.choice function used.
    elif len(args.frequencies) == 0:
        if codes is None: codes = get_codes(records)
        args.frequencies = [np.mean(codes == c) for c in args.alphabet]
    else: # given as a list of frequencies
        assert len(args.frequencies) == len(args.alphabet), "len(freqs)=" + str(len(args.frequencies)) + "\tlen(alph)=" + str(len(args.alphabet))


    with as_handle(args.outfile, "w") as outfile:
        for record in records:
            record.seq = randomize(record.seq, args.alphabet, args.frequencies, args.replace)
            bio.write(record.seq, record.description, outfile) 
 

if __name__ == '__main__':
    main(get_parser().parse_args())

