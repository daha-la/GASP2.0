#!/usr/bin/env python3
import argparse
import pandas as pd
from degnutil import argument_parsing as argue, bio_util as bio, pandas_util as panda
from Bio.File import as_handle
import warnings
# The line "col = np.where(key_or_index == header)[0]" in column_index causes a future warning.
# The future behavior was the intended behavior, however it fails comparison of 0 == array(['a', 'b']) due to difference in types.
# The result is the same since it returns False. It would only be a problem if we had e.g. 0 == array(['a', 'b', 0]),
# but the array will always be read as string.
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_parser():
    parser = argparse.ArgumentParser(description="Convert a table to fasta.")
    argue.inoutfile(parser)
    argue.delimiter(parser)
    argue.header(parser)
    argue.fieldsep(parser)
    argue.kvsep(parser)
    parser.add_argument("-i", "--id", dest="ids", default=[0], nargs="*", help="Name or index of sequence ID column(s). Default=[0].")
    parser.add_argument("-s", "--seq", default=-1, help="Sequence column number or name. Default=-1 (last column).")
    parser.add_argument("-a", "--annotations", action="store_true", help="Write key-value pairs to fasta headers from all other table columns.")

    return parser


def parse_args(args):
    return argue.parse_inoutfile(argue.parse_delimiter(args))


def main(args):
    args = parse_args(args)
    
    df = panda.read_pandas(args.infile, args.delimiter, args.header)
    args.ids = panda.colID2name(df, args.ids)
    args.seq = panda.colID2name(df, args.seq)
    annotation_names = [c for c in df.columns if c not in args.ids + [args.seq]] if args.annotations else []

    with as_handle(args.outfile, 'w') as outfile:
        for _, row in df.iterrows():
            ids = [row[_id] for _id in args.ids]
            id_str = " ".join([str(_id) for _id in ids if not pd.isna(_id)])
            annotations = [f"{a}{args.kvsep}{row[a]}" for a in annotation_names]
            description = args.fieldsep.join([id_str] + annotations)
            bio.write(row[args.seq], description, outfile)

    

if __name__ == '__main__':
    main(get_parser().parse_args())

