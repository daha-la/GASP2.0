#!/usr/bin/env python3
import argparse
import math
import numpy as np
from degnutil import argument_parsing as argue, motif_util as mot


def get_parser():
    parser = argparse.ArgumentParser(description="Align multiple motifs (MMA) with restrictions to alignment length etc. Inspired by tomtom.")
    argue.inoutfile(parser, help="MEME minimal format.")
    parser.add_argument("-b", "--bins", default=100,
                        help="How many bins to divide each Probability Density Function into. Higher numbers cost more but gives more precise p-values.")
    parser.add_argument("-l", "--length", 
            help="Length of alignment. All motifs have to fit inside this length. Trailing parts are trimmed and unaccounted for in alignment scoring.")
    
    return parser


def parse_args(args):
    return argue.parse_inoutfile(args)


def QT2XY(Q, T):
    """
    Get two 3D arrays for comparison between each column in query and target.
    :param Q: list of matrices or matrix, dim 0 is position, dim 1 is letter
    :param T: list of matrices or matrix, dim 0 is position, dim 1 is letter
    :return: X, Y
    """
    # converts list of matrices to single matrix (cat along dim 0), and does nothing to a matrix.
    Q = np.vstack(Q)
    T = np.vstack(T)
    X = np.asarray([Q for _ in range(len(T))])
    Y = np.asarray([T for _ in range(len(Q))])
    X = X.swapaxes(0, 1)
    assert X.shape == Y.shape
    return X, Y


def ED(X, Y):
    """
    Euclidean distance score.
    Reduce over the last axis alone, 
    which means if 1d you get a scalar score, if 2d you get a vector of scores and if 3d you get a matrix of scores.
    :param X: array of values, e.g. frequencies for each letter.
    :param Y: array of values, e.g. frequencies for each letter.
    :return: -euclidean distance
    """
    X, Y = np.asarray(X), np.asarray(Y)
    return -np.sqrt(np.sum((X - Y)**2, axis=-1))


def center_medians(Γ):
    """
    Center medians for each row.
    from shift_all_pairwise_scores in tomtom.c
    :param Γ: matrix (2D array)
    :return: matrix
    """
    # medians are made into a column vector so we make sure that they are subtracted for each element along the rows.
    return Γ - np.median(Γ, axis=1)[:, np.newaxis]


def integer_score(Γ, bins):
    """
    Scale scores into range [0,bins]
    from scale_score_matrix in pssm.c
    :param Γ: scalar or array of floats
    :param bins: max number
    :return: array of ints, score offset, score scale
    """
    low, high = Γ.min(), Γ.max()
    if low == high: low -= 1
    offset = low = math.floor(low)
    scale = math.floor(bins / (high-low))
    return np.round((Γ - offset) * scale).astype(int), offset, scale

# 
# 
# 
# def get_score(Γ, offsets, window, score_offset, score_scale):
#     """
#     Get the integer score for a specific overlap configuration
#     from compute_overlap_score in tomtom.c
#     :param Γ: 
#     :param offsets: the offsets each array has relative to start of the alignment window 
#     :param window: length of alignment window 
#     :param score_offset: 
#     :param score_scale: 
#     :return: integer score, int length of overlap
#     """
#     wQ, wT = Γ_QT.shape
#     if config <= wQ: Q_start, T_start = wQ - config, 0
#     else: Q_start, T_start = 0, config - wQ
#     scores = [Γ_QT[i, j] for i, j in zip(range(Q_start, wQ), range(T_start, wT))]
#     return sum(scores), len(scores)


def score2overlap_score(scores, overlaps, score_offset, score_scale):
    """
    Before score comparison tomtom makes a correction to the score that was found simply by summing Γ entries.
    Here's the correction if we want to use it.
    :return: scoring including overlap consideration.
    """
    return np.round(scores + overlaps * score_offset * score_scale)


def align_motifs(motifs, length, bins=100):
    """
    Align difficult motifs, considering that we want
    - high information content from each motif within the alignment
    - high info content letters should match the same letter in other motifs of high info content
    - high info content letters should fall back to matching other high info letters even if it's not the same letters 
    :param motifs: dict of unaligned motifs potentially of varying lengths. 
    Each array is a matrix where dim 0 is position in sequence, and dim 1 is for letter in alphabet
    :param length: length of alignment.
    :param bins: number of possible integer scores
    :return: aligned motifs each of length "length"
    """
    arrays = motifs.values()
    T = np.vstack(arrays)
    
    Γ = ED(*QT2XY(T, T))
    Γ = center_medians(Γ)
    Γ, score_offset, score_scale = integer_score(Γ, bins)
    
    w_arrays = [len(array) for array in arrays]
    # optimal_scores, optimal_overlaps, optimal_offsets = \
    #     zip(*(optimal_pairwise(Γ[:, target_stop-w_target : target_stop], score_offset, score_scale) 
    #           for w_target, target_stop in zip(w_arrays, np.cumsum(w_arrays))))
    
    return arrays


def main(args):
    args = parse_args(args)
    memes = mot.read_meme(args.infile)
    memes["matrices"] = align_motifs(memes["matrices"], args.length, args.bins)
    # mot.write_meme(args.outfile, **memes)


if __name__ == '__main__':
    main(get_parser().parse_args())

