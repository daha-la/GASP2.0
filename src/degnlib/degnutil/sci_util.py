#!/usr/bin/env python3
import numpy as np
from scipy.stats import truncnorm
from sklearn.utils.extmath import stable_cumsum
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy.integrate.odepack import ODEintWarning
from scipy import interp
import warnings
from degnutil import array_util as ar


"""
This script is for scipy and other science related functions.
"""


def normal_trunc(lower=-1, upper=1, mean=None, std=None, size=1):
    if mean is None: mean = (lower + upper) / 2
    if std is None: std = (upper - lower) / 6
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


def roc_micro(y_true, y_score, drop_intermediate=True):
    """
    A micro average ROC calculation for averaging two boolean classifications.
    we use sign of y_true as -1=class 1, 0=no class, 1=class 2 
    :param y_true: 
    :param y_score: 
    :param drop_intermediate: 
    :return: fpr, tpr, thres
    """
    y_true = np.concatenate((y_true < 0, y_true > 0))
    y_score = np.concatenate((-y_score, y_score))
    return roc_curve(y_true, y_score, drop_intermediate=drop_intermediate)


def precision_recall_micro(y_true, y_score):
    """
    A micro average ROC calculation for averaging two boolean classifications.
    we use sign of y_true as -1=class 1, 0=no class, 1=class 2 
    :param y_true: 
    :param y_score: 
    :return: fpr, tpr, thres
    """
    y_true = np.concatenate((y_true < 0, y_true > 0))
    y_score = np.concatenate((-y_score, y_score))
    return precision_recall_curve(y_true, y_score)


def roc_macro(y_true, y_score, drop_intermediate=True):
    """
    A macro average ROC calculation for averaging two boolean classifications.
    we use sign of y_true as -1=class 1, 0=no class, 1=class 2 
    :param y_true: 
    :param y_score: 
    :param drop_intermediate: 
    :return: fpr, tpr
    """
    fpr_neg, tpr_neg, _ = roc_curve(y_true < 0, -y_score, drop_intermediate=drop_intermediate)
    fpr_pos, tpr_pos, _ = roc_curve(y_true > 0, y_score, drop_intermediate=drop_intermediate)
    
    fpr = np.unique(np.concatenate((fpr_neg, fpr_pos)))
    # interpolate all ROC curves at these points and average
    tpr = (interp(fpr, fpr_neg, tpr_neg) + interp(fpr, fpr_pos, tpr_pos)) / 2
    return fpr, tpr


def precision_recall_macro(y_true, y_score):
    """
    A macro average ROC calculation for averaging two boolean classifications.
    we use sign of y_true as -1=class 1, 0=no class, 1=class 2 
    :param y_true: 
    :param y_score: 
    :return: fpr, tpr
    """
    prec_neg, recall_neg, _ = precision_recall_curve(y_true < 0, -y_score)
    prec_pos, recall_pos, _ = precision_recall_curve(y_true > 0, y_score)
    
    prec = np.unique(np.concatenate((prec_neg, prec_pos)))
    recall = (interp(prec, prec_neg, recall_neg) + interp(prec, prec_pos, recall_pos)) / 2
    
    return prec, recall


def f1(prec, recall):
    """
    Get the F1 score of a classifier.
    https://en.wikipedia.org/wiki/F1_score
    :param prec: 1D precisions
    :param recall: 1D recalls
    :return: f1-score 
    """
    return ar.nandiv(2 * prec * recall, prec + recall).mean()


def roc_curve_sign(y_true, y_score, drop_intermediate=True):
    """
    A version of roc curve calculation (lists tpr, fpr) 
    where instead of having a boolean classification problem and changing a threshold to see how many tp and fp we get,
    we have two categories, decided by sign (-1: first class, 0: no class, 1: second class).
    So we trust that a score with absolute value above the threshold has a sign that is the accurate classification.

    Can e.g. be used for edges in a network, where 0 is a negative example, -1, or 1 is a positive example meaning there is an edge.
    A true positive is finding an edge with a specific sign, a false positive is finding an edge but it is not actually there.
    A false negative is not finding an edge which is there and true negative is not indicating any edge while it being true.
    :param y_true: 
    :param y_score: 
    :param drop_intermediate:
    :return: 
    """
    n_pos = sum(y_true != 0)
    n_neg = sum(y_true == 0)
    fps, tps, thresholds = sign_clf_curve(y_true, y_score)

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless")
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / n_neg

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless")
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / n_pos

    return fpr, tpr, thresholds


def sign_clf_curve(y_true, y_score):
    """
    from sklearn.metrics.ranking import _binary_clf_curve
    :param y_true: float 1D where sign is used
    :param y_score: float 1D
    :return: fps, tps, absolute threshold values
    """
    y_true = np.sign(y_true)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(abs(y_score), kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(abs(y_score)))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    y_score_sign = np.sign(y_score)
    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum((y_score_sign != 0) & (y_true == y_score_sign))[threshold_idxs]
    fps = stable_cumsum((y_score_sign != 0) & (y_true != y_score_sign))[threshold_idxs]
    return fps, tps, abs(y_score[threshold_idxs])



def odeint_filter(function, *args):
    """
    Filter odint call so only allow successful ones.
    Implement assertions in function to filter out those as unsuccessful.
    Return None if unsuccessful, otherwise return output of the given function
    :param function: some function using the odeint of scipy
    :param args: args to give to the function
    :return: function output or None
    """
    def excess_work(msg):
        """
        Return if the ODEintWarning is the excess work type
        :param msg: ODEintWarning
        :return: bool
        """
        return "Excess work" in str(msg)
    
    with warnings.catch_warnings(record=True) as ws:
        # warnings.simplefilter("error")
        try:
            print("trying")
            out = function(*args)
        except AssertionError:
            print("assertion error. give up")
            return
        
        print("looking for recorded warnings")
        for w in ws:
            print("here's a recorded warning:", w)
            # bad configuration of network except for excess work that might just be due to slow convergence
            if w.category == ODEintWarning and not excess_work(w.message):
                print("there was an ODE int warning recorded that was NOT excess work")
                return

    return out



