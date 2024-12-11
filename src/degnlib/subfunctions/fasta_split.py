#!/usr/bin/env python3
import argparse
from Bio import SeqIO
from Bio.File import as_handle
from degnutil import argument_parsing as argue, bio_util as bio, pandas_util as panda
from degnutil.input_output import log
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description="Split sequences at specific position(s) or between specific domains.")
    argue.inoutfile(parser)
    argue.delimiter(parser)
    argue.header(parser)
    argue.fieldsep(parser)
    argue.kvsep(parser)
    argue.quiet(parser)
    argue.verbose(parser)
    parser.add_argument("-t", "--table", required=True,
                        help="Table with a text column of sequence IDs and a column with numbers to indicate places to split. "
                             "If there are two numbers found per row then they are understood as ranges for domains that should be separated. "
                             "In that case a sequence is split at the half-way point between two domains if both domains overlap with the sequence.")
    parser.add_argument("-0", "--origin", default="origin",
                        help="Name of the description field with the info. "
                             'By default look for description field with name "origin", otherwise fallback to using 1.')
    parser.add_argument("-a", "--annotation",
                        help="Seq annotation that is matched against the text column in the -t/--table file. "
                             "Default=seq id. If the table has a header (-H/--header) the default is the text column name.")
    parser.add_argument("-1", "--one-indexed", action="store_true", 
                        help="Set flag if all locations in -t/--table are given as if sequences are 1-indexed. "
                             "This flag allows -0/--origin to refer to the true origin that will still be updated.")
    parser.add_argument("-m", "--minus", action="store_true",
                        help="Allow negative indexing to count from the end of sequences. "
                             "This argument is ignored unless -1/--one-indexed is also set."
                             "Default is ignoring negative split locations.")
    
    return parser
    

def parse_args(args):
    return argue.parse_inoutfile(argue.parse_delimiter(args))


def split_sequence(record, splits, origin:str="origin"):
    """
    Split sequence at the given positions in "splits".
    Splits before the index (e.g. split=2 means A|BC if 1-indexed and AB|C if 0-indexed)
    :param record: SeqRecord object where description has been parsed to annotations
    :param splits: [int position,...] 0-indexed, relative to start of sequence string, ignoring origin annotation
    :param origin: str key for origin field to edit
    :return: [SeqRecord seq,...]
    """
    records = [record]
    _origin = 0
    for location in np.sort(splits):
        if 0 <= location - _origin < len(records[-1]):
            records, _origin = _split_records(records, location, _origin, origin)
    return records


def split_sequence_domains(record, intervals, origin:str="origin"):
    """
    Split sequence halfway between overlapping domains, marked by ranges.
    :param record: SeqRecord object where description has been parsed to annotations
    :param intervals: [[int start, int end],...] 0-indexed, relative to start of sequence string, ignoring origin annotation
    :param origin: str key for origin field to edit
    :return: [SeqRecord seq,...]
    """
    # sort based on the earliest start location. Shouldn't matter if sorting by start or end location unless intervals overlap.
    intervals = np.asarray(intervals)
    intervals = intervals[np.argsort(intervals[:,0]),:]
    
    records = [record]
    _origin = 0
    for i in range(len(intervals)-1):
        # end of one interval and beginning of the next should both be inside the sequence
        if 0 <= intervals[i][1] - _origin < len(records[-1]) and 0 <= intervals[i+1][0] - _origin < len(records[-1]):
            # halfway point
            location = int((intervals[i][1] + intervals[i+1][0]) / 2)
            records, _origin = _split_records(records, location, _origin, origin)
    return records


def _split_records(records, location:int, _origin, origin:str="origin"):
    """
    Helper function.
    Splits the last record in "records" and shifts the "_origin"
    :param records: [SeqRecord,...]
    :param location: int split location
    :param _origin: int index of first position in last record of "records"
    :param origin: str annotation key
    :return: [SeqRecord,...], _origin
    """
    # the split distance from start of the last record in records
    dist = location - _origin
    left, right = bio.slice_seq(records[-1], stop=dist), bio.slice_seq(records[-1], start=dist)
    # move origin to start of the rightmost sequence
    _origin += dist
    # update origin annotation if present
    try: right.annotations[origin] += dist
    except KeyError: pass
    records = records[:-1] + [left, right]
    return records, _origin


def main(args):
    args = parse_args(args)

    keys, integers, table_annotation = panda.read_id_integers(args.table, args.delimiter, args.header)
    for v in integers: assert len(v) <= 2, "More than two numbers found in table row(s)."
    for v in integers: assert len(v) > 0, "No numbers found in table row(s)."
    # make mapping from ID to a list of ranges or split positions.
    # splittings = {str ID:[[int start, int end],...],...} OR
    # splittings = {str ID:[[int split],...],...}
    splittings = {k: [] for k in keys}
    is_ranges = all(len(i) == 2 for i in integers)
    if is_ranges:
        log.info("Splitting halfway between ranges.")
        for k, v in zip(keys, integers): splittings[k].append(v)
    else:
        for k, v in zip(keys, integers): splittings[k].extend(v)
    # remove empty and convert to numpy
    splittings = {k:np.asarray(vs) for k,vs in splittings.items() if len(vs) > 0}
    assert len(splittings) > 0, "No split locations or ranges found in table."
    
    records = list(bio.yield_parsed_records(SeqIO.parse(args.infile, "fasta"), args.fieldsep, args.kvsep))
    
    # setting args.annotation directly takes priority over what is found in the table
    if args.annotation is not None:
        annotations = [bio.get_annotation(r, args.annotation) for r in records]
    else:
        # see if we are supposed to use the annotation name in the table or it is just some unimportant naming
        annotations = [bio.get_annotation(r, table_annotation, default="") for r in records]
        if all(np.char.asarray(annotations) == ""):
            # the annotation name from the table was no use, fallback to use id
            annotations = [bio.get_annotation(r) for r in records]
            log.info("Using record ID to map to table.")
        else:
            log.info("Using record annotation {} to map to table.".format(table_annotation))
    
    
    split_func = split_sequence_domains if is_ranges else split_sequence
    
    def _to_0_index(splitting, record):
        """
        :param splitting: array
        :param record: SeqRecord
        :return: array
        """
        if not args.minus:
            if args.one_indexed: return splitting - 1
            # make each location relative to origin annotation value
            else: return splitting - record.annotations.get(args.origin, 1)
        else:
            splitting[splitting < 0] += len(record)
            log.info("{}:\t{}".format(record.description, splitting))
            return splitting
    
    n_splits = 0
    with as_handle(args.outfile, 'w') as outfile:
        for record, annotation in zip(records, annotations):
            try: splitting = splittings[annotation]
            except KeyError: bio.write(str(record.seq), record.description, outfile)
            else:
                for seq in bio.yield_deparsed_seqs(split_func(record, _to_0_index(splitting, record), args.origin)):
                    bio.write(str(seq.seq), seq.description, outfile)
                    n_splits += 1
                n_splits -= 1  # one less split than number of seqs written
    
    log.info("Number of splits:\t" + str(n_splits))


if __name__ == '__main__':
    main(get_parser().parse_args())

