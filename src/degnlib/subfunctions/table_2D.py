#!/usr/bin/env python3
import sys
import argparse
from degnutil import argument_parsing as argue
from Bio import File

def get_parser():
    parser = argparse.ArgumentParser(
        description='Take a table that has multiple fields in each cell and make it into a "2D" table with a single field per cell. '
                    'If a given row has n fields in one of the cells, then other cells in that row should have 1 or n fields.')
    argue.inoutfile(parser)
    argue.delimiter(parser)
    parser.add_argument("-D", "--sep2", dest="delimiter2", default=",",
                        help="Secondary delimiter. Separates fields inside each cell.")
    return parser

def parse_args(args):
    return argue.parse_inoutfile(argue.parse_delimiter(args))

def main(args):
    args = parse_args(args)
    outdelim = args.delimiter if args.delimiter is not None else "\t"
    
    with File.as_handle(args.infile) as infile, File.as_handle(args.outfile, 'w') as outfile:
        # it doesn't matter is file has header or not for this
        for line in infile:
            cells = line.strip().split(args.delimiter)
            cells = [cell.split(args.delimiter2) for cell in cells]
            n_fields = max(len(cell) for cell in cells)
            for i_cell in range(len(cells)):
                if len(cells[i_cell]) == 1: cells[i_cell] *= n_fields
                else: assert len(cells[i_cell]) == n_fields, "different numbers of fields found in cells in a row"
            for i_field in range(n_fields):
                outfile.write(outdelim.join([cell[i_field] for cell in cells]) + "\n")


if __name__ == '__main__':
    main(get_parser().parse_args())

