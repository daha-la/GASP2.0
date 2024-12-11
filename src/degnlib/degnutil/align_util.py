#!/usr/bin/env python3
import numpy as np

def get_pos2cons(seq):
    """
    Get index relative to consensus within a sequence.
    "abC-DeFg" -> array([-2, -1, 0, 1, 2, 2.5, 3, 4])
    :param seq: string sequence where consensus is uppercase letters and dashes.
    :return: vector of same length as seq where each position indicates relative location from the perspective of consensus.
    """
    seq = np.asarray(list(seq))
    cons_idx = np.char.isupper(seq) | (seq == '-')
    # cumsum will increase the index for every consensus position.
    # We want to be zero-indexed so we subtract 1 which will make the first match equal to 0 and earlier positions == -1
    cons_ind = np.cumsum(cons_idx) - 1
    cons_max = cons_ind[-1]  # max value is at the very end

    # set insertions within consensus to have float value.
    cons_ind = np.asarray(cons_ind, dtype=float)
    for v in range(cons_max):
        cons_ind[cons_ind == v] = np.linspace(v, v + 1, (cons_ind == v).sum(), endpoint=False)

    # set early positions to have different values depending on distance from first cons position
    cons_ind[cons_ind == -1] = np.arange(-(cons_ind == -1).sum(), 0)
    # similar at the other end
    cons_ind[cons_ind == cons_max] = cons_max + np.arange((cons_ind == cons_max).sum())

    return cons_ind
# unit test
assert np.all(get_pos2cons("abC-DeFg")  == np.asarray([-2, -1, 0, 1, 2, 2.5, 3, 4]))
assert np.all(get_pos2cons("-abC-DeFg") == np.asarray([0, 1/3, 2/3, 1, 2, 3, 3.5, 4, 5]))
assert np.all(get_pos2cons("-abC-D.Fg") == np.asarray([0, 1/3, 2/3, 1, 2, 3, 3.5, 4, 5]))


def get_cons_pos(seqs, pos):
    """
    Get position relative to consensus where multiple consensus domains can be given from the same protein.
    :param seqs: dict mapping from protein position to sequence with consensus highlighted in uppercase and dashes.
    :param pos: int position relative to sequence to be converted to relative to consensus start
    :return: converted positions that are relative to consensus. Same size as pos, np.nan if no match.
    """
    cons_pos = np.repeat(np.nan, len(seqs))
    for i, (offset, seq) in enumerate(seqs.items()):
        pos2cons = get_pos2cons(seq)
        p = pos - offset
        if 0 <= p < len(pos2cons):
            cons_pos[i] = pos2cons[p]

    return cons_pos








