#!/usr/bin/env python3
import argparse
from Bio.File import as_handle
from degnutil import argument_parsing as argue, bio_util as bio


def get_parser():
    parser = argparse.ArgumentParser(description="Convert selex to table. Currently without header.")
    argue.inoutfile(parser)
    argue.delimiter_tab(parser)
    return parser
    

def parse_args(args):
    return argue.parse_inoutfile(args)


def main(args):
    args = parse_args(args)
    with as_handle(args.outfile, 'w') as outfile:
        for seq, name in zip(*bio.read_selex(args.infile)):
            outfile.write(name + args.delimiter + seq + '\n')
    

if __name__ == '__main__':
    main(get_parser().parse_args())

