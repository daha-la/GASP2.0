#!/usr/bin/env python3
import argparse
from degnutil import argument_parsing as argue
from Bio import SeqIO
from Bio.File import as_handle

def get_parser():
    parser = argparse.ArgumentParser(description="Print length of each sequence.")
    argue.infiles(parser)
    argue.outfile(parser)
    return parser
    
def main(args):
    with as_handle(args.outfile, 'w') as outfile:
        for infile in args.infiles:
            for record in SeqIO.parse(infile, "fasta"):
                outfile.write(str(len(record.seq)) + '\n')

if __name__ == '__main__':
    parsed_args = get_parser().parse_args()
    main(parsed_args)

