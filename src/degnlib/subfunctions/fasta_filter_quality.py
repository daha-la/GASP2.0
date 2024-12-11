#!/usr/bin/env python3
import argparse
from Bio import SeqIO
from Bio.File import as_handle
from degnutil.string_util import count_uppercase
from degnutil import bio_util as bio

from degnutil import argument_parsing as argue

def get_parser():
    parser = argparse.ArgumentParser(description="Filter out sequences of poor quality, stemming from bad alignment.")
    argue.inoutfile(parser)
    parser.add_argument("-t", "--thres", dest="threshold", required=True, 
                        help="Minimum number of uppercase letters in sequence or minimum fraction. ")
    
    return parser
    

def parse_args(args):
    argue.parse_inoutfile(args)
    try: args.threshold = int(args.threshold)
    except ValueError: args.threshold = float(args.threshold)
    return args


def main(args):
    args = parse_args(args)
    
    records = SeqIO.parse(args.infile, "fasta")
    
    with as_handle(args.outfile, 'w') as outfile:
        if type(args.threshold) is float:
            for record in records:
                if count_uppercase(record) >= args.threshold * len(record):
                    bio.write(str(record.seq), record.description, outfile)
        else:
            for record in records:
                if count_uppercase(record) >= args.threshold:
                    bio.write(str(record.seq), record.description, outfile)
           

if __name__ == '__main__':
    main(get_parser().parse_args())

