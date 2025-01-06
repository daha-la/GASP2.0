#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from degnutil import argument_parsing as argue, bio_util as bio
from Bio import File, SeqIO


def get_parser():
    parser = argparse.ArgumentParser(description="Convert fasta to table.")
    argue.inoutfile(parser)
    argue.delimiter_tab(parser)
    argue.fieldsep(parser)
    argue.kvsep(parser)
    parser.add_argument("-i", "--id", help='Name of sequence ID column. Default="id" (if -H/--header or -s/--seq is given).')
    parser.add_argument("-s", "--seq", help='Sequence column name. Default="seq" (if -H/--header or -i/--id is given).')
    parser.add_argument("-H", "--header", action="store_true", help="Flag if table should have a header. "
                                                                    "If -i or -s contains header names, it will not be necessary to set this flag.")
    parser.add_argument("-a", "--annotations", action="store_true", help="Extract key-value paired annotations from headers.")
    return parser

def parse_args(args):
    if args.id or args.seq: args.header = True
    if args.header or args.annotations:
        if args.id is None: args.id = "id"
        if args.seq is None: args.seq = "seq"
    args = argue.parse_delimiter(args)
    return argue.parse_inoutfile(args)


def main(args):
    args = parse_args(args)
    
    with File.as_handle(args.outfile, 'w') as outfile:
        if not args.annotations:
            if args.header:
                outfile.write(args.delimiter.join([args.id, args.seq]) + '\n')
            for record in SeqIO.parse(args.infile, "fasta"):
                outfile.write(args.delimiter.join([record.description, str(record.seq)]) + '\n')
        else:
            records = list(bio.yield_parsed_records(SeqIO.parse(args.infile, "fasta"), args.fieldsep, args.kvsep))
            df = pd.DataFrame({args.id: [bio.get_record_id(r, args.id) for r in records]})
            anno_names = {k for r in records for k in r.annotations}
            for name in anno_names: df[name] = [r.annotations.get(name, "") for r in records]
            df[args.seq] = [str(r.seq) for r in records]
            df.to_csv(outfile, sep=args.delimiter, index=False)


if __name__ == '__main__':
    main(get_parser().parse_args())

