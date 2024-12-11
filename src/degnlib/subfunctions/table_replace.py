#!/usr/bin/env python3
import argparse
from degnutil import argument_parsing as argue
from degnutil.pandas_util import read_pandas, write_pandas


def get_parser():
    parser = argparse.ArgumentParser(description="Replace a substring inside cells. E.g. useful to remove newlines inside cells.")
    argue.inoutfile(parser)
    argue.delimiter(parser)
    argue.header(parser)
    parser.add_argument("-r", "--replace", required=True, nargs="+",
                        help='Provide a value for what should be replaced, and a second for what it should be replaced by. '
                             'Provide a single value to remove the value.')
    return parser


def parse_args(args):
    if len(args.replace) == 1: args.replace.append('')
    elif len(args.replace) > 2: raise argparse.ArgumentTypeError("Max 2 values can be given to -r/--replace.")
    return argue.parse_inoutfile(argue.parse_delimiter(args))


def main(args):
    args = parse_args(args)
    table = read_pandas(args.infile, sep=args.delimiter, header=args.header)
    table.replace(args.replace[0], args.replace[1], inplace=True, regex=True)
    write_pandas(args.outfile, table, args.delimiter)


if __name__ == '__main__':
    main(get_parser().parse_args())

