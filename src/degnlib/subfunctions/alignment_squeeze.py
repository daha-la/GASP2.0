#!/usr/bin/env python3
import argparse
import numpy as np
from Bio import SeqIO
from Bio.File import as_handle
from degnutil import argument_parsing as argue, bio_util as bio
from degnutil.input_output import log

"""
EXAMPLE
input:
>
.....ab---C.DE--FGHI--.---jkl....
output:
>
.......-abC.DE--FGHIjk.l--.......
output with -u/--upper:
>
.......-ABC.DE--FGHIJK.L--.......
"""


def get_parser():
    parser = argparse.ArgumentParser(
        description="hmmalign will produce alignments where the terminal unmatched letters are moved outside consensus region. "
                    "This function moves them into the consensus, next to the other letters where they actually are in the sequence.")
    argue.inoutfile(parser, 
                    help="Fasta file made from hmmalign selex then converted to fasta. "
                         "Sequences has '.' for outside consensus, '-' for gap inside consensus, uppercase letter for consensus match, "
                         "and lowercase for either insertion or the terminal unmatched letters that will be moved with this script.")
    argue.quiet(parser)
    parser.add_argument("-u", "--upper", action="store_true", 
                        help="Set flag to convert tail letters that end up inside the consensus to uppercase. "
                             "This makes can make it more clear where consensus starts and ends after the move.")

    return parser


def parse_args(args):
    return argue.parse_inoutfile(args)


def squeeze(seq, to_upper=False):
    """
    Move lowercase letters in the start close to the first uppercase letters 
    and the lowercase letters at the end close to the last uppercase letters.
    They can move onto '-' but not '.'. They will leave '.' behind when moved.
    Resulting sequence should have the same length.
    :param seq: SeqRecord, Seq, str
    :param to_upper: bool. Should letters be converted to uppercase if they are moved onto locations where '-' were before?
    :return: str
    """
    seq = np.char.asarray(list(seq))
    try: up_start, up_end = np.where(seq.isupper())[0][[0,-1]]
    except IndexError:
        # no uppercase letters, it happens if it is e.g. the consensus sequence
        log.info("Sequence with no uppercase letters unchanged.")
        return ''.join(seq)
    
    before_up = np.zeros(len(seq), dtype=bool)
    after_up = np.zeros(len(seq), dtype=bool)
    before_up[:up_start] = True
    after_up[up_end+1:] = True
    
    is_low = seq.islower()
    low_start = seq[is_low & before_up]
    low_end = seq[is_low & after_up]
    
    # remove the lowercase fragments
    seq[is_low & before_up] = '.'
    seq[is_low & after_up] = '.'
    
    # This is only correct AFTER inserting the '.' at low_start_idx and low_end_idx
    cons_start, cons_end = np.where(seq != '.')[0][[0, -1]]
    
    dash_start = (seq == '-') & before_up
    dash_end = (seq == '-') & after_up
    # describe the locations where the lowercase fragments can be moved to
    low_start_idx = np.copy(dash_start)
    low_end_idx = np.copy(dash_end)
    # add all the '.' before and after consensus
    low_start_idx[:cons_start] = True
    low_end_idx[cons_end+1:] = True
    low_start_idx = np.where(low_start_idx)[0]
    low_end_idx = np.where(low_end_idx)[0]
    
    seq[low_start_idx[len(low_start_idx)-len(low_start):]] = low_start
    seq[low_end_idx[:len(low_end)]] = low_end
    
    if to_upper:
        seq[dash_start] = seq[dash_start].upper()
        seq[dash_end] = seq[dash_end].upper()
    
    return ''.join(seq)


def main(args):
    args = parse_args(args)
    
    with as_handle(args.outfile, 'w') as outfile:
        for record in SeqIO.parse(args.infile, "fasta"):
            bio.write(squeeze(record, args.upper), record.description, outfile)


if __name__ == '__main__':
    main(get_parser().parse_args())
