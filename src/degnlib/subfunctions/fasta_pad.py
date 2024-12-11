#!/usr/bin/env python3
import argparse
from Bio import SeqIO
from Bio.File import as_handle
from degnutil import argument_parsing as argue, bio_util as bio, read_write as rw, string_util as st
from degnutil.path_util import isfile


def get_parser():
    parser = argparse.ArgumentParser(description="Pad something to one or both ends of sequences.")
    argue.inoutfile(parser)
    argue.delimiter(parser)
    parser.add_argument("-H", "--header", action="store_true", help="Set flag if a given -l/--left or -r/--right table has header.")
    parser.add_argument("-l", "--left",
            help="Characters to pad at the start of sequences or interger to indicate repeated padding of character given by -g/--gap. Can also be given as table.")
    parser.add_argument("-r", "--right", nargs="?", const=True,
            help="Characters to pad at the end of sequences or interger to indicate repeated padding of character given by -g/--gap. Can also be given as table. \
                    If arg is given without a value, the sequences are padded to match in length.")
    parser.add_argument("-g", "--gap", default="-", help="String to pad multiple times if -r/--left or -r/--right is given as integers.")
    
    return parser
    

def parse_args(args):

    if args.left is not None:
        if isfile(args.left):
            args.left = rw.read_vector(args.left, args.delimiter, args.header)
        else:
            args.left = st.parse_number(args.left)
            if isinstance(args.left, int): args.left *= args.gap
        
    if args.right not in [None, True]:
        if isfile(args.right):
            args.right = rw.read_vector(args.right, args.delimiter, args.header)
        else:
            args.right = st.parse_number(args.right)
            if isinstance(args.right, int): args.right *= args.gap

    return argue.parse_inoutfile(args)


def main(args):
    args = parse_args(args)
    
    records = list(SeqIO.parse(args.infile, "fasta"))

    # first offset the beginning of sequences.
    if isinstance(args.left, str):
        for record in records:
            record.seq = args.left + record.seq
    
    elif isinstance(args.left, dict):
        for record in records:
            left = args.left[record.description]
            try: record.seq = left + record.seq
            except TypeError: record.seq = left * args.gap + record.seq

    elif isinstance(args.left, list):
        for record, left in zip(records, args.left):
            try: record.seq = left + record.seq
            except TypeError: record.seq = left * args.gap + record.seq

    # pad at the end of sequences
    if isinstance(args.right, str):
        for record in records:
            record.seq += args.right

    elif isinstance(args.right, dict):
        for record in records:
            right = args.right[record.description]
            try: record.seq += right
            except TypeError: record += right * args.gap

    elif isinstance(args.right, list):
        for record, right in zip(records, args.right):
            try: record.seq += right
            except TypeError: record += right * args.gap

    elif args.right is True:
        # pad to get the same length
        length = max(len(r.seq) for r in records)
        for record in records:
            record.seq += (length - len(record.seq)) * args.gap
    
    with as_handle(args.outfile, "w") as outfile:
        for record in records:
            bio.write(record.seq, record.description, outfile) 
 

if __name__ == '__main__':
    main(get_parser().parse_args())

