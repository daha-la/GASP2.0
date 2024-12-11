#!/usr/bin/env python3
import argparse
import math
import numpy as np
import pandas as pd
from degnutil import argument_parsing as argue, motif_util as mot, pandas_util as panda


def get_parser():
    parser = argparse.ArgumentParser(
        description="Replicate of tomtom. Get optimal offsets for pairwise alignments of motifs and p-values. "
                    "All vs All. Assumes all motif alphabets are identical. Reverse compliment and incomplete scores not implemented.")
    argue.inoutfile(parser, help="MEME minimal infile format and table output.")
    argue.delimiter(parser)
    parser.add_argument("-b", "--bins", default=100, 
                        help="How many bins to divide each Probability Density Function into. Higher numbers cost more but gives more precise p-values.")
    
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


def optimal_pairwise(Γ_QT, score_offset, score_scale):
    """
    
    :param Γ_QT: region of Γ containing columns for a single query and rows for a single target.
    :param score_offset:
    :param score_scale:
    :return: optimal score, overlap, offset
    """
    scores, overlaps, offsets = zip(*(get_overlap_score(Γ_QT, config, score_offset, score_scale) 
                                      for config in range(1, sum(Γ_QT.shape))))
    scores = np.asarray(scores)
    overlaps = np.asarray(overlaps)
    best  = scores.max() == scores
    best &= (overlaps[best].max() == overlaps)
    best = np.where(best)[0][0]
    return scores[best], overlaps[best], offsets[best]


def get_overlap_score(Γ_QT, config, score_offset, score_scale):
    """
    Get the integer score for a specific overlap configuration
    from compute_overlap_score in tomtom.c
    :param Γ_QT: 
    :param config: 
    :param score_offset: 
    :param score_scale: 
    :return: integer score, overlap length, offset of query first position relative to target first position 
    """
    wQ, wT = Γ_QT.shape
    if config <= wQ: Q_start, T_start = wQ - config, 0
    else: Q_start, T_start = 0, config - wQ
    scores = [Γ_QT[i, j] for i, j in zip(range(Q_start, wQ), range(T_start, wT))]
    score, overlap = sum(scores), len(scores)
    assert overlap > 0
    # UK: undo the offset when median shifted scores are used 
    return np.round(score + overlap * score_offset * score_scale), overlap, config - wQ


def get_PMFs(Γ, bins):
    """
    Get probability mass function histogram for all possible ranges within a query.
    from get_pv_lookup_new in tomtom.c
    :param Γ: matrix with integer scores with shape len(query) x len(all targets)
    :param bins: 
    :return: Probability Mass Function values in 3D array, indexing is [start index in query, end index in query, bin in PMF]
    """
    wQ, wT = Γ.shape
    
    # I think this becomes triangular
    PMFs = np.zeros([wQ, wQ, wQ*bins + 1])
    
    # make pdf for each potential section of query (start anywhere and end with any length contained inside the query)
    for start in range(wQ-1, 0-1, -1):
        PDF_last = np.zeros(wQ*bins + 1)
        PDF_last[0] = 1
        # end = start + i is the last query column in the overlap (so "i+1" is the overlap size)
        for i in range(wQ - start):
            for j in range(wT):
                s = Γ[start + i, j]
                PMFs[start, start+i, s:s+i*bins+1] += PDF_last[:i*bins+1] / wT

            PDF_last = PMFs[start, start+i, :]

    return PMFs


def get_max_align_p(PMFs, w_query, w_targets, score_offset, score_scale):
    """
    Calculates the p-values of the shifted scores by finding the distribution of the maximal shifted alignments score for each possible target motif length
    from reference_matrix in tomtom.c
    :param PMFs: 
    :param w_query: 
    :param w_targets: 
    :param score_offset: 
    :param score_scale: 
    :return: 
    """
    # pmf size if offsets are peeled off. Note that for an positive quantile shifted scores offset is always >= 0
    PMFs_ext = np.zeros([PMFs.shape[0], PMFs.shape[1], math.ceil(PMFs.shape[2] - (w_query-1) * score_offset * score_scale)])

    for start in range(w_query):
        for end in range(start, w_query):
            overlap = end - start + 1
            offset_correction = round((w_query - overlap) * score_offset * score_scale)
            assert offset_correction <= 0
            PMFs_ext[start, end, -offset_correction : PMFs.shape[2]-offset_correction] = PMFs[start, end, :]

    #// compute the probability mass function (pmf) of the best alignment shifted score for each target motif width
    max_align_p = {}
    for w_target in np.unique(w_targets):
        PMF = np.zeros(PMFs_ext.shape[2])
        PMF[0] = 1
        
        for i_align in range(w_query + w_target - 1):
            if i_align < w_query:  # find the start and end columns of this alignment
                start_col = w_query - i_align - 1
                end_col = min(w_query, start_col+w_target) - 1
            else:
                start_col = 0
                end_col = min(w_query, w_target+w_query - i_align - 1) - 1
            
            PMF_ext = PMFs_ext[start_col, end_col, :]
            CDF_ext = PMF_ext.cumsum()
            CDF     = PMF.cumsum()
            
            # next is the heart of the calculation of the PMF
            PMF[0] *= PMF_ext[0]
            PMF[1:] = PMF[1:] * (CDF_ext[:-1] + PMF_ext[1:]) + PMF_ext[1:] * CDF[:-1]
    
        max_align_p[w_target] = PMF[::-1].cumsum()[::-1]
    return max_align_p


def score2p(scores, w_query, w_targets, max_align_p, score_offset, score_scale):
    scores = np.round(np.asarray(scores) - w_query * score_offset * score_scale)
    return np.asarray([max_align_p[w_target][score] for score, w_target in zip(scores, w_targets)])


def align_query(query, targets, bins):
    """
    :param query: unaligned motif array potentially of varying lengths. 
    :param targets: unaligned motif arrays potentially of varying lengths. 
    Each array is a matrix where dim 0 is position in sequence, and dim 1 is for letter in alphabet
    :param bins: number of possible integer scores
    :return: p-values, optimal offsets, optimal overlaps
    """
    Q, T = np.asarray(query), np.vstack(targets)
    wQ = len(Q)

    Γ, score_offset, score_scale = integer_score(center_medians(ED(*QT2XY(Q, T))), bins)

    w_targets = [len(target) for target in targets]
    optimal_scores, optimal_overlaps, optimal_offsets = \
        zip(*(optimal_pairwise(Γ[:, target_stop - w_target: target_stop], score_offset, score_scale)
              for w_target, target_stop in zip(w_targets, np.cumsum(w_targets))))

    max_align_p = get_max_align_p(get_PMFs(Γ, bins), wQ, w_targets, score_offset, score_scale)
    ps = score2p(optimal_scores, wQ, w_targets, max_align_p, score_offset, score_scale)
    
    return ps, optimal_offsets, optimal_overlaps


def align_all(arrays, bins=100):
    """
    :param arrays: unaligned motifs potentially of varying lengths. 
    Each array is a matrix where dim 0 is position in sequence, and dim 1 is for letter in alphabet
    :param bins: number of possible integer scores
    :return: pandas array with columns Query, Target, p-value, E-value, Offset, Overlap
    """
    df = {"Query":[], "Target":[], "p-value":[], "E-value":[], "Offset":[], "Overlap":[]}
    
    targets = list(arrays.values())
    for query_name, query in arrays.items():  # all vs. all
        ps, offsets, overlaps = align_query(query, targets, bins)
        df["Query"].extend(query_name for _ in targets)
        df["Target"].extend(arrays.keys())
        df["p-value"].extend(ps)
        df["Offset"].extend(offsets)
        df["Overlap"].extend(overlaps)
    
    df["E-value"] = np.asarray(df["p-value"]) * len(targets)
    return pd.DataFrame(df)


def main(args):
    args = parse_args(args)
    memes = mot.read_meme(args.infile)
    df = align_all(memes["matrices"], args.bins)
    panda.write_pandas(args.outfile, df, args.delimiter)


if __name__ == '__main__':
    main(get_parser().parse_args())

