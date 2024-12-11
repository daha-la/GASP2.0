#!/usr/bin/env python3
import argparse
from Bio import SeqIO
from Bio.File import as_handle
from degnutil import argument_parsing as argue, bio_util as bio


def get_parser():
    parser = argparse.ArgumentParser(description="Convert sequences from 3-letter codes to 1-letter codes.")
    argue.inoutfile(parser)
    
    return parser
    

def parse_args(args):
    return argue.parse_inoutfile(args)


def main(args):
    args = parse_args(args)
    
    with as_handle(args.outfile, 'w') as outfile:
        for record in SeqIO.parse(args.infile, "fasta"):
            record = bio.three2one(record)
            bio.write(record.seq, record.description, outfile)
           

if __name__ == '__main__':
    main(get_parser().parse_args())

