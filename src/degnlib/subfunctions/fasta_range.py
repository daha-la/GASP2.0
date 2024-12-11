#!/usr/bin/env python3
import argparse
import os
import re

import pandas as pd
from Bio import SeqIO
from Bio.File import as_handle
from degnutil import argument_parsing as argue, bio_util as bio
from degnutil.bio_util import get_record_id, subslice_1letter
from degnutil.input_output import log
from copy import deepcopy


def get_parser():
    parser = argparse.ArgumentParser(description="Return ranges from each sequence.")
    argue.inoutfile(parser)
    argue.fieldsep(parser)
    argue.kvsep(parser)
    argue.delimiter(parser)
    parser.add_argument("-H", "--header", action="store_true", help="Set flag if a given -t/--table table has header.")
    parser.add_argument("-t", "--table", help="Provide ranges in a table with either columns -i/--id and -r/--range or an id column followed by a range column.")
    parser.add_argument("-r", "--range", nargs="+",
                              help="A range given as a start and end number, "
                                   "separated by any non-number character(s) (e.g. ... or - or a space). "
                                   "Can also be a single position. "
                                   "If a filename is given to -t/--table, then -r/--range can be set to column name that holds ranges. Default in that case is \"range\".")
    parser.add_argument("-0", "--origin", default="origin",
                        help="Index of the first position in sequences or name of the description field with the info. "
                             'By default look for description field with name "origin", otherwise fallback to using 1.')
    parser.add_argument("-a", "--pad", default=0, type=int, help="Add extra positions at both ends of the range. Can be negative. Default=0.")
    parser.add_argument("-D", "--description", action="store_true", help="Write the range to the end of sequence IDs (format: ID 123-456).")
    parser.add_argument("-p", "--per-range", action="store_true", 
                        help="Write sequence for each range provided in range file. "
                             "Default is per sequence in the input fasta file. "
                             "Ignored if range file is not provided.")
    parser.add_argument("-f", "--force", action="store_true", 
                        help="By default an error is raised when using range match from file and mapping is missing or range in table is missing. "
                             "Set flag to ignore and skip sequences with missing range (or write empty sequences, see -k/--keep).")
    parser.add_argument("-k", "--keep", action="store_true", help="Set flag to keep sequences with missing ranges resulting in empty sequences.")
    parser.add_argument("-i", "--id", default="id", help='Identifier or annotation from records to match between record and table entries. Default=id.')
    argue.quiet(parser)
    
    return parser

def parse_args(args):
    args = argue.parse_inoutfile(args)

    if args.table is not None:
        assert os.path.isfile(args.table)
        if len(args.range) == 0: args.range = "range"
        elif len(args.range) == 1: args.range, = args.range
        else: raise argparse.ArgumentTypeError("Too many range column names given.")
    else:
        for r in args.range: assert not os.path.isfile(r)
        args.range = parse_range(*args.range)

    try: args.origin = int(args.origin)
    except ValueError: pass

    return args


def parse_positions(string):
    """
    Find ints and single letter codes followed by int
    :param string:
    :return: list
    """
    rs = re.findall("[A-Z]?[0-9]+", string)
    for i, r in enumerate(rs):
        try: rs[i] = int(r)
        except ValueError: pass
    return rs

def parse_range(*string):
    """
    Get range from string, e.g.
    "4" -> (4, 4)
    "4..5" -> (4, 5)
    "4-5" -> (4, 5)
    "K4...P9" -> ("K4", "P9")
    ("K4", "P9") -> ("K4", "P9")
    ("K4", "filename") -> ValueError
    :param string: (str,) or (str, str)
    :return: tuple range
    """
    if len(string) == 1:
        rs = parse_positions(string[0])
        if len(rs) == 1: return rs[0], rs[0]
        if len(rs) == 2: return tuple(rs)
        raise ValueError(f"No range found in \"{string[0]}\".")
    elif len(string) == 2:
        rs = parse_positions(string[0]), parse_positions(string[1])
        assert len(rs[0]) == 1 and len(rs[1]) == 1, "Incorrect number of positions given by 2 args"
        return rs[0] + rs[1]
    else:
        raise ValueError("Wrong number of strings given for parsing a range.")

def read_ranges(fname, header, id_col=None, range_col=None):
    df = pd.read_table(fname, header=0 if header else None)
    if header: ids, ranges = df[id_col], df[range_col]
    else: ids, ranges = df[0], df[1]
    ranges = [parse_range(s) for s in ranges]
    return ids, ranges


def _not_found_warning(notfound):
    log.warning("Records not found:\n" + "\n".join(notfound))


def main(args):
    args = parse_args(args)
    if type(args.origin) == int: get_origin = lambda r: args.origin
    else: get_origin = lambda r: r.annotations.get(args.origin, 1)
    
    records = list(bio.yield_parsed_records(SeqIO.parse(args.infile, "fasta"), args.fieldsep, args.kvsep))

    if args.table is not None:
        range_ids, ranges = read_ranges(args.table, args.header, args.id, args.range)
        
        if args.per_range:
            # select records based on ids for each range selection
            _n = len(records)
            id2record = dict(zip([get_record_id(record, args.id) for record in records], records))
            if len(id2record) < _n:
                if args.force: log.error("Record IDs are not unique, number of potential records to match reduced.")
                else: raise ValueError("Records has to have unique identifiers to use -p/--per-range.")
                
            records = [deepcopy(id2record.get(k, None)) for k in range_ids]
            if any(r is None for r in records):
                if not args.force: raise KeyError("No match found among records for some ids from table.")
                _not_found_warning(set(range_ids) - id2record.keys())
                if not args.keep:
                    # reduce collections so they match
                    range_ids, ranges = zip(*((i, r) for i, r in zip(range_ids, ranges) if i in id2record))
                    records = list(filter(None, records))
        else:
            # convert ranges to lookup without duplicates, assume there are non
            id2ranges = dict(zip(range_ids, ranges))
            # select ranges based on ids for each record
            record_ids = [get_record_id(record, args.id) for record in records]
            try: ranges = [(i, id2ranges[i]) for i in record_ids]
            except KeyError:
                if not args.force: raise
                _not_found_warning(set(record_ids) - id2ranges.keys())
                if args.keep:
                    range_ids, ranges = zip(*((i, id2ranges.get(i, None)) for i in record_ids))
                else:
                    # reduce collections so they match
                    record_ids, records = zip(*((i, rec) for i, rec in zip(record_ids, records) if i in id2ranges))
                    range_ids, ranges = zip(*((i, id2ranges[i]) for i in record_ids))
        
        assert len(records) == len(ranges)
        for record, range_id, _range in zip(records, range_ids, ranges):
            if record is None or _range is None: continue
            assert get_record_id(record, args.id) == range_id, f"{get_record_id(record, args.id)} != {range_id}"
            try:
                record.annotations["slice"] = subslice_1letter(record.seq, *_range, get_origin(record), args.pad)
            except AssertionError:
                if not args.force: raise
                log.warning("Single letter code mismatch.")
                record.annotations["slice"] = slice(0, 0)
    else:
        for record in records:
            try:
                record.annotations["slice"] = subslice_1letter(record.seq, args.range[0], args.range[1], get_origin(record), args.pad)
            except AssertionError:
                if not args.force: raise
                log.warning("Single letter code mismatch.")
                record.annotations["slice"] = slice(0, 0)

    # change sequences based on the slice annotation we just added
    for record in records:
        if record is not None:
            if "slice" in record.annotations:
                record.seq = record.seq[record.annotations["slice"]]
            else:
                record.seq = record.seq[0:0]

    if args.description:
        for record in records:
            if record is not None:
                if "slice" in record.annotations:
                    origin = get_origin(record)
                    start = record.annotations["slice"].start + origin
                    end   = record.annotations["slice"].stop  + origin - 1
                    record.description += args.fieldsep + "%d-%d" % (start, end)
    
    with as_handle(args.outfile) as outfile:
        for record in records:
            if record is not None:
                bio.write(record.seq, record.description, outfile)
            elif args.keep:
                outfile.write(">\n")  # empty record
           

if __name__ == '__main__':
    main(get_parser().parse_args())

