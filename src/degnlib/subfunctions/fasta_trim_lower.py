#!/usr/bin/env python3
import argparse
import numpy as np
from Bio import SeqIO
from Bio.File import as_handle
from degnutil import argument_parsing as argue, bio_util as bio


def get_parser():
    parser = argparse.ArgumentParser(description="Trim lowercase letters from sequences. "
                                                 "Written originally for 'hmmalign --outformat A2M' output in order to trim what is outside the profileHMM consensus sequence. "
                                                 "hmmalign can also do this with argument --trim, however we want to know how much was trimmed (see -0/--origin).")
    argue.inoutfile(parser)
    parser.add_argument("-0", "--origin", help="Field name in header for index of first letter in sequence. If provided, then the field will be updated to reflect the trim.")
    parser.add_argument("-d", "--sep", dest="fieldsep", default=' ', help="Sequence id field separator")
    parser.add_argument("-D", "--sep2", default='=', dest="kvsep", help="Sequence id key/value pair separator")
    
    return parser
    

def parse_args(args):
    return argue.parse_inoutfile(args)


def consensus_range_A2M(seq):
    """
    
    :param seq: np.char.array
    :return: range
    """
    # A2M format shows consensus in uppercase and pushes anything outside out padding with '-', 
    # so we use ~islower instead of isupper
    start, end = np.where(~seq.islower())[0][[0,-1]]
    # +1 to change end from index of last position to the stop parameter of a range
    return range(start, end+1)


def trim_lowercase(seq):
    """
    Lowercase letters trimmed from either end of a sequence.
    Also get the amount of letters trimmed from the start.
    :param seq: string, Seq, or SeqRecord
    :return: (seq, int start)
    """
    _range = consensus_range_A2M(np.char.asarray(list(seq)))
    return seq[_range], _range.start


def trim_lowercase_record(record, origin, fieldsep, kvsep):
    """
    Trim lowercase letters from a sequence.
    Also get a description of the record where a field for the start index of the sequence has been updated from the trim.
    :param record: SeqRecord.
    :param origin: str. key for field to update.
    :param fieldsep: str. delimiter for fields
    :param kvsep: str. delimiter between a key and the value.
    :return: (str sequence, str record description)
    """
    seq, ntrim = trim_lowercase(str(record.seq))
    description = record.description.split(fieldsep)
    for i, field in enumerate(description):
        kv = field.split(kvsep)
        if len(kv) == 2 and kv[0] == origin:
            description[i] = origin + kvsep + str(int(kv[1]) + ntrim)
    
    return seq, fieldsep.join(description)


def main(args):
    args = parse_args(args)
    
    records = SeqIO.parse(args.infile, "fasta")
    
    with as_handle(args.outfile, 'w') as outfile:
        if args.origin is None:
            for record in records:
                bio.write(trim_lowercase(str(record.seq))[0], record.description, outfile)
        else:
            for record in records:
                bio.write(*trim_lowercase_record(record, args.origin, args.fieldsep, args.kvsep), outfile)


if __name__ == '__main__':
    main(get_parser().parse_args())

