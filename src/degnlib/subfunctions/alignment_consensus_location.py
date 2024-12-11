#!/usr/bin/env python3
import argparse
from pandas.api.types import is_integer_dtype
import numpy as np
import pandas as pd
from Bio import SeqIO
from degnutil import argument_parsing as argue, bio_util as bio, pandas_util as panda


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert locations relative to unaligned sequences into locations relative to consensus origin (0-indexed).")
    argue.inoutfile(parser, help="Table where first text column is keys, and first integer column is locations.")
    argue.delimiter(parser)
    argue.header(parser)
    parser.add_argument("-s", "--seqs", dest="sequences", required=True,
                        help="Aligned sequences. Written for hmmalign --outformat selex converted to fasta in mind: "
                             "First entry should be a consensus sequence with x for consensus and '.' for the rest.")
    parser.add_argument("-0", "--origin", default="origin",
                        help='Field name in seq description for index of first letter in sequence. Default="origin".')
    parser.add_argument("-a", "--annotation",
                        help="Seq annotation that is matched against the text column in the infile. "
                             "Default=seq id. "
                             "If infile has a header (-H/--header) the default is the text column name, "
                             "but only if sequences are found to have annotations with that name.")
    parser.add_argument("-i", "--no-insertions", action="store_true", 
                        help="Set flag if locations should be given relative to consensus without insertions.")
    parser.add_argument("-g", "--gap", default="-.", help="Gap characters that makes the difference between the aligned and unaligned.")
    argue.quiet(parser)

    return parser


def parse_args(args):
    return argue.parse_inoutfile(argue.parse_delimiter(args))


def aligned_location(location, records, origins, gap='-'):
    """
    Get the location in an alignment, given a location relative to unaligned record origin.
    :param location: location relative to unaligned record
    :param records: array of SeqRecord(s). Potential record(s) where the location could be in. 
    They should all be different regions of the same overall sequence.
    :param origins: array of ints. Origin(s) of the sequence(s)
    :param gap: str. gap characters
    :return: location in aligned sequence, SeqRecord it aligned to
    """
    for i in np.argsort(origins):
        seq = np.char.array(list(records[i]))
        is_gap = np.zeros(len(seq), dtype=bool)
        for g in gap: is_gap = is_gap | (seq == g)
        un2aligned = np.arange(len(seq))[~is_gap]
        if location - origins[i] in range(len(un2aligned)):
            return un2aligned[location - origins[i]], records[i]
    return np.nan, None


def main(args):
    args = parse_args(args)

    ## read table
    table = panda.read_pandas(args.infile, args.delimiter, args.header)
    key_col, location_col = None, None
    for col, dtype in zip(table, table.dtypes):
        if location_col is None and is_integer_dtype(dtype): location_col = col
        elif key_col is None: key_col = col
    assert key_col is not None, "No text column found."
    assert location_col is not None, "No number column found."
    
    ## read sequences
    records = list(bio.yield_parsed_records(SeqIO.parse(args.sequences, "fasta")))
    consensus = np.char.asarray(list(records[0]))
    assert all((consensus == '.') | (consensus == 'x')), "Unrecognized consensus sequence format."
    cons_start, cons_end = np.where(consensus == 'x')[0][[0,-1]]
    # filter out seq if origin annotation is missing
    records = [r for r in records[1:] if args.origin in r.annotations]
    assert len(records) > 0, "No sequences found with origin annotation."
    
    # manually setting args.key takes priority over what is found in the file.
    if args.annotation is not None:
        rec_keys = np.char.asarray([bio.get_annotation(r, args.annotation) for r in records])
    else:
        try: rec_keys = np.char.asarray([bio.get_annotation(r, key_col) for r in records])
        # fall back to record id
        except KeyError: rec_keys = np.char.asarray([r.id for r in records])

    align_locs = np.repeat(np.nan, len(table))
    table["description"] = ""
    for row in table.itertuples():
        loc = getattr(row, location_col)
        idx = np.where(getattr(row, key_col) == rec_keys)[0]
        loc_recs = [records[i] for i in idx]
        loc_origins = [r.annotations[args.origin] for r in loc_recs]
        align_locs[row.Index], loc_rec = aligned_location(loc, loc_recs, loc_origins, args.gap)
        # we also wish to update the key column so we can keep track of which range of a sequence the location was found in
        if loc_rec is not None:
            table.loc[row.Index, "description"] = loc_rec.description

    # bool index for insertion
    insertions = np.zeros(len(consensus), dtype=bool)
    insertions[cons_start:cons_end + 1] = (consensus == '.')[cons_start:cons_end + 1]  # +1 makes end -> stop
    # which locations are insertions?
    _insertions = np.where(insertions)[0]
    table["insert"] = [loc in _insertions for loc in align_locs]

    if args.no_insertions:
        w2wo_insert = np.cumsum(~insertions) - 1  # -1 since the first value is true so it becomes 1-indexed
        for i, loc in enumerate(align_locs):
            if not np.isnan(loc):
                align_locs[i] = w2wo_insert[int(loc)]
        

    # relative to consensus start
    table["cons_loc"] = align_locs - cons_start
    table["cons_loc"] = table["cons_loc"].astype(pd.Int64Dtype())  # convert to int type that can contain NA
    panda.write_pandas(args.outfile, table, args.delimiter)



if __name__ == '__main__':
    main(get_parser().parse_args())
