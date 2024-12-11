#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from degnutil import argument_parsing as argue, motif_util as mot, read_write as rw

def get_parser():
    parser = argparse.ArgumentParser(description="Shift motifs by some offset by padding zeros upstream. "
                                                 "Also pads zeros downstream so motifs match in length. ")
    argue.inoutfile(parser)
    parser.add_argument("-o", "--offset", dest="offsets", required=True, 
                        help="Table with column for motif names and another with integer offsets.")
    argue.delimiter(parser)
    argue.header(parser)
    
    return parser


def parse_args(args):
    return argue.parse_inoutfile(args)


def shift(arr, offset:int):
    """
    Pad zero rows at the beginning of arr.
    :param arr: numpy or pandas array
    :param offset: amount of rows to add.
    :return: numpy or pandas array
    """
    padding = np.zeros((offset, arr.shape[1]))
    if isinstance(arr, pd.DataFrame):
        padding = pd.DataFrame(padding, columns=arr.columns)
        return pd.concat((padding, arr), ignore_index=True)
    else:
        return np.concatenate((padding, arr))


def pad(arr, length:int):
    """
    Pad zero rows at the end of arr to increase its length to "length"
    :param arr: numpy or pandas array
    :param length: int
    :return: numpy or pandas array
    """
    padding = np.zeros((length - arr.shape[0], arr.shape[1]))
    if isinstance(arr, pd.DataFrame):
        padding = pd.DataFrame(padding, columns=arr.columns)
        return pd.concat((arr, padding), ignore_index=True)
    else:
        return np.concatenate((arr, padding))


def main(args):
    args = parse_args(args)
    
    offsets = rw.read_vector(args.offsets, args.delimiter, args.header)
    offsets = dict(offsets)
    memes = mot.read_meme(args.infile)
    
    for name, motif in memes["matrices"].items():
        memes["matrices"][name] = shift(motif, offsets[name])

    length = max(v.shape[0] for v in memes["matrices"].values())
    for name, motif in memes["matrices"].items():
        memes["matrices"][name] = pad(motif, length)

    mot.write_meme(args.outfile, **memes)


if __name__ == '__main__':
    main(get_parser().parse_args())

