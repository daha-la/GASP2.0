#!/usr/bin/env python3
import math
import numpy as np

"""
This script is for utilities related to numpy arrays.
"""


def argmin(x, n):
    """
    Get indexes of top n from x. n == 1 is the same as np.argmin
    :param x: collection
    :param n: number of top values returned
    :return: unsorted indexes for top n values in x
    """
    return np.argpartition(x, n)[:n]


def argmax(x, n):
    """
    Get indexes of top n from x. n == 1 is the same as np.argmax
    :param x: collection
    :param n: number of top values returned
    :return: unsorted indexes for top n values in x
    """
    return np.argpartition(x, -n)[-n:]



def ndim(arr):
    """
    np.ndim that finds max number of dimensions if the array is jagged
    :param arr: array or list or string or number or many other types
    :return: int
    """
    try: arr = np.asarray(arr)
    except ValueError: pass  # most likely jagged
    else:
        # dtype is object if the array is mixed types or if it is jagged
        # use != instead of not is since it is numpy's own object type
        if arr.dtype != object: return arr.ndim
    return max(ndim(a) for a in arr) + 1


def ascol(arr1d):
    return arr1d.reshape((-1, 1))


def l1(arr):
    return abs(arr).sum()


def l2(arr):
    return (arr ** 2).sum()


def mm_norm(arr, K_M=0.5):
    """
    Michaelis-Menten inspired l-norm.
    :param arr: array or tensor
    :param K_M: decides steepness. It holds michaelis_menten(K_M) = 0.5
    :return: float
    """
    return michaelis_menten(arr, K_M).sum()


def michaelis_menten(arr, K_M=0.5):
    """
    Michaelis-Menten function.
    :param arr: array or tensor
    :param K_M: decides steepness. It holds michaelis_menten(K_M) = 0.5
    :return: array or tensor of same shape as arr
    """
    arr = abs(arr)
    return arr / (K_M + arr)



def logFC(wt, obs):
    """
    Get the logFC = log2(ko/wt) for some measurement values, e.g. x, y, phi, xyphi, etc.
    :param wt: array with wildtype measurements. The last dimension separates the variables.
    wildtype array should have 1 less dimensions than obs array.
    :param obs: array with observed measurements. The last dimension separates the variables.
    With three dimensions they should be for condition/experiment, time, node respectively.
    :return: logFC values for the observed measurements relative to wildtype.
    """
    return np.log2(obs / wt)


def where(logical):
    """
    To find the integer index of a logical where you know there is only one answer, 
    or you are only interested in the first instance.
    :param logical: boolean array
    :return: int index
    """
    return np.nonzero(logical)[0][0]


def invwhere(index, length, dtype=bool):
    """
    Inverse of the where/nonzero function.
    Convert an integer index to a boolean/logical index.
    :param index: int or list of ints
    :param length: length of the output
    :param dtype: can be int or float to get 1s and 0s instead of True and False
    :return: 
    """
    out = np.zeros(length, dtype=dtype)
    out[index] = 1
    return out


def normal_masked(matrix):
    """Create a matrix with normal random values using param matrix as mask."""
    matrix = np.asarray(matrix, dtype=bool)
    std = 1. / math.sqrt(matrix.shape[0] * matrix.shape[1])
    out = np.random.normal(scale=std, size=matrix.shape)
    out[~matrix] = 0
    return np.asarray(out, dtype='float32')


def noise(n, m=None):
    """
    Simply a matrix of random normal values where std decreases with size.
    :param n: size of axis 0 of the array
    :param m: size of axis 1 of the array, default to n
    :return: array with gaussian distributed random values
    """
    if m is None: m = n
    return fl(np.random.normal(scale=1. / np.sqrt(n * m), size=(n, m)))


def inv(arr):
    """
    inverse matrix that works for difficult matrices.
    See unit test for example.
    :param arr: numpy array
    :return: numpy array inverse of arr
    """
    I = np.eye(arr.shape[1], dtype='float64')
    return np.linalg.solve(arr.astype('float32'), I).astype('float64')


def cov(arr):
    """
    Fix a small difference from symmetry that can occur in the covariance func.
    :param arr: 
    :return: 
    """
    return symmetric(np.cov(arr))


def corr(arr, arr2=None):
    """
    Fix that corrcoef automatically makes float64.
    Fix a small difference from symmetry that can occur in the correlation estimate.
    :param arr: 
    :return: 
    """
    cor = np.corrcoef(arr, arr2).astype('float32')
    return symmetric(cor)


def precision(arr):
    """
    Get the estimated precision matrix given observations.
    Fix the small difference from symmetry that can occur in the inverse estimate.
    WARNING: this result will be a float64 since it is big. 
    It can sometimes give incorrect dot on another float64 for some reason.
    :param arr: observations of variables. 
    Each row is a variable, each column is observations across variables.
    :return: estimated precision matrix. 
    """
    return inv(cov(arr))


def partial_corr(arr):
    """
    Get the estimated partial correlation matrix given observations.
    :param arr: observations of variables. 
    Each row is a variable, each column is observations across variables.
    :return: estimated partial correlation matrix. 
    """
    return _partial_corr(cov(arr))
    

def _partial_corr(cov):
    """
    Get the estimated partial correlation matrix given covariance.
    :param cov: covariance square matrix. 
    :return: estimated partial correlation matrix. 
    """
    prec = inv(cov)
    prec_ii = prec_jj = np.diag(prec)
    # following formula is meaningless for the diagonal
    pcor = -prec / np.sqrt(np.outer(prec_ii, prec_jj))
    np.fill_diagonal(pcor, 1)
    return pcor


def partial_correlation_1st(rAB, rAC, rBC):
    """
    1st order partial correlation. 
    There is a difference of sign between this and the matrix method for some reason.
    :param rAB: correlation
    :param rAC: correlation
    :param rBC: correlation
    :return: rAB.C correlation
    """
    return (rAB - rAC*rBC) / np.sqrt((1 - rAC**2) * (1 - rBC**2))


def symmetric(arr):
    """
    Force an array to be symmetrical by averaging.
    :param arr: array that ideally is close to symmetric.
    :return: symmetric array
    """
    return (arr + arr.T) / 2


def fill_except_diag(a, fill):
    """
    Fill entries in an array that is not in the diagonal.
    :param a: array to insert values in.
    :param fill: scalar or array to insert.
    :return: array with filled entries for chaining.
    """
    # first assume matrix
    try: fill = fill.T.flat
    # otherwise it's a scalar
    except AttributeError: pass
    a.T[~np.eye(a.shape[1], dtype=bool)] = fill
    return a


def random_mask(maximum, shape, empty_diag=False):
    """
    Create a 2d array of randomly distributed 1s and 0s 
    where each column has between 1 and "maximum" number of 1s.
    :param maximum: int maximum number of 1s in any given column
    :param shape: shape of array
    :param empty_diag: make sure the diagonal is left out and is empty
    :return: 2d numpy array
    """
    def _random_mask(maximum, shape):
        arr = np.zeros(shape)
        ns = np.random.randint(1, maximum + 1, shape[1])
        for i, n in enumerate(ns):
            arr[:n,i] = 1
            np.random.shuffle(arr[:,i])
        return arr
    
    if not empty_diag: return _random_mask(maximum, shape)
    # leave out diagonal from design
    fill = _random_mask(maximum, (shape[0]-1, shape[1]))
    # add empty diagonal elements
    return fill_except_diag(np.zeros(shape), fill)


def fl(arr):
    return arr.astype("float32", copy=False)


def binary(num, width=None):
    """
    Convert an integer to binary where the binary number is represented as a list of booleans.
    :param num: integer to convert
    :param width: optional width
    :return: 1D bool array representation of the number
    """
    return np.asarray(list(np.binary_repr(num, width)), dtype=int).astype(bool)


def nonzero_mean(arr, axis=None):
    """
    Mean across the nonzero values in an array.
    :param arr: numpy array
    :param axis: reduction along this axis
    :return: 
    """
    nominator = arr.sum(axis=axis)
    denominator = (arr != 0).sum(axis=axis)
    try: denominator[denominator == 0] = 1
    except TypeError:
        if denominator == 0:
            denominator = 1
    return nominator / denominator


def nandiv(nominator, denominator, nan=0):
    """
    numpy division where divisions returning nan instead returns a non-nan value.
    :param nominator: 
    :param denominator: 
    :param nan: value to get instead of nan
    :return: nominator / denominator
    """
    idx = denominator == 0
    denominator[idx] = 1
    out = nominator / denominator
    out[idx] = nan
    return out



