#!/usr/bin/env python3
import argparse
from degnutil import argument_parsing as argue
from degnutil.pandas_util import read_pandas, write_pandas
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(description="Copy non-empty cells into empty cells below in order to fill all empty cells. "
                                                 "WARNING: pandas have had issues with reading double quotes inside a field which I could not solve by changing reading params. "
                                                 "One solution is tr them away and back after this call.")
    argue.inoutfile(parser)
    argue.delimiter(parser)
    parser.add_argument("-c", "--column", dest="columns", nargs="*", 
                        help="Column name(s) or number(s) to change. Default=all columns.")
    argue.header(parser)
    return parser

def parse_args(args):
    return argue.parse_inoutfile(argue.parse_delimiter(args))

def main(args):
    args = parse_args(args)
    table = read_pandas(args.infile, sep=args.delimiter, header=args.header)
    
    if args.columns is None: args.columns = table.columns
    for col in args.columns:
        if col not in table.columns:
            try: col = table.columns[int(col)]
            except ValueError:
                raise argparse.ArgumentTypeError("-c/--column given unknown column")
        
        last_value = None
        for i in range(len(table)):
            cell = table.loc[i, col]
            if pd.isna(cell): table.loc[i, col] = last_value
            else: last_value = cell
    
    write_pandas(args.outfile, table, args.delimiter, header=args.header)


if __name__ == '__main__':
    main(get_parser().parse_args())

