#!/usr/bin/env python3
import argparse
from Bio.File import as_handle
from pathlib import Path
import numpy as np
from degnutil import argument_parsing as argue, bio_util as bio

def get_parser():
    parser = argparse.ArgumentParser(description="Count number of occurrences of a code (letter) in each position.")
    argue.inoutfile(parser)
    argue.delimiter(parser, help="Delimiter for reading -g/--group table. Output delimiters not customizable.")
    parser.add_argument("-c", "--code", default="-", help="Code (letter) to count. Default is gap.")
    parser.add_argument("-g", "--group", nargs="?", const=True, 
                        help="By default counting is among all sequences. Set flag to perform count for each unique seq description. "
                             "Count for specific groupings by providing a file with groups of ids on each line.")
    return parser

def parse_args(args):
    if args.group not in [None, True]:
        args.group = [l.split(args.delimiter) for l in Path(args.group).read_text().split("\n")]
    return argue.parse_inoutfile(args)


def count_letter(strings, letter):
    """
    Count occurrences of letter in each position across strings.
    :param strings: 
    :param letter:
    :return: vector of ints
    """
    return sum(np.asarray(list(s)) == letter for s in strings)


def main(args):
    args = parse_args(args)
    seqs, names = bio.read(args.infile)

    with as_handle(args.outfile, "w") as outfile:
        # count among all
        if args.group is None:
            counts = count_letter(seqs, args.code)
            outfile.write("\n".join(map(str,counts))+"\n")
        
        # count among sequences with identical names
        elif args.group is True:
            seqs, names = np.asarray(seqs), np.asarray(names)
            for name in np.unique(names):
                counts = count_letter(seqs[names == name], args.code)
                outfile.write(name + "\t" + " ".join(map(str, counts)) + "\n")

        # count among sequences within each specified group
        else:
            seqs, names = np.asarray(seqs), np.asarray(names)
            for group in args.group:
                g_seqs = np.concatenate([seqs[names == n] for n in group])
                counts = count_letter(g_seqs, args.code)
                try: outfile.write(" ".join(group) + "\t" + " ".join(map(str, counts)) + "\n")
                except TypeError:
                    # if there are no matches for a group we get counts == 0 which is not iterable, hence the error.
                    outfile.write(" ".join(group) + "\tnan\n")


if __name__ == '__main__':
    main(get_parser().parse_args())

