#!/usr/bin/env python3
import time
import math
import numpy as np
import pandas as pd
import theano as th
from theano.tensor.extra_ops import fill_diagonal as tensor_fill_diagonal
from theano import sparse as sp, tensor as T
from theano.tensor import slinalg as tsla
import scipy.sparse as spsp
from degnlib.degnutil import array_util as ar, math_util as ma, gradient_descent as gd, neural_network as neu

"""
This script is for utilities related to theano.
"""

def std_across(arrays):
    """
    Get the mean standard deviation across arrays, 
    only getting standard deviations between arrays and not within arrays.
    :param arrays: list of theano arrays
    :return: float std
    """
    return stack(arrays).std(axis=0).mean().eval().item()


def stack(tensors):
    """
    Stack tensors, e.g. 2D tensors of same shape to make a 3D tensor.
    :param tensors: theano tensors or theano sparse tensors
    :return: theano tensor.
    """
    try: return T.stack(tensors)
    except TypeError: pass
    for i in range(len(tensors)):
        try: tensors[i] = tensors[i].toarray()
        except AttributeError: pass
    return T.stack(tensors)


def shared(matrix, name):
    """Convert an array to a shared sparse or dense theano matrix depending on all values being allowed.
    """
    assert matrix is not None and name
    matrix = np.asarray(matrix, dtype='float32')
    if np.sum(matrix == 0) == 0: return th.shared(matrix, name)
    # if there are values set to zero, it is assumed they should NOT be changed.
    # this is accomplished by using a sparse matrix, and structure_dot.
    # VERY IMPORTANT!! has to be csc, csr diverges the cost and is just very weird
    matrix = spsp.csc_matrix(matrix, dtype='float32')
    return sp.shared(matrix, name)


def variable_to_array(variable):
    """
    Convert a theano variable to a numpy array
    :param variable: theano variable
    :return: dense numpy array
    """
    array = variable.eval()
    if not is_sparse(array): return array
    return array.toarray()  # don't use todense


def col(vec):
    """
    Force a theano vector to be column vector
    :param vec: 1d vector
    :return: 2d column vector
    """
    return vec.dimshuffle(0, 'x')


def super_dot(As, B):
    """
    Theano helper function
    Do matrix multiplication (dot) as A . B but for each column in B a different A is used. 
    :param As: list of As, assumed to have length of B shape 1 (width)
    :param B: 
    :return:
    """
    tensors = [col(As[i].dot(B[:, i])) for i in range(len(As))]
    return T.concatenate(tensors, axis=1)


def is_sparse(array):
    return hasattr(array, "toarray")


def dot(a, b):
    """
    Theano dot that allows both sparse and dense for a but not flexible for b.
    :param a: 
    :param b: 
    :return: 
    """
    return sp.structured_dot(a, b) if is_sparse(a) else T.dot(a, b)


def thinv(arr, symmetric=False):
    """
    inverse matrix that works for difficult matrices.
    See unit test for example.
    :param arr: numpy array or theano tensor
    :param symmetric: set flag to true for method optimized for symmetric arr.
    :return: theano tensor inverse of arr
    """
    I = T.eye(arr.shape[1], dtype='float64')
    A_structure = 'symmetric' if symmetric else 'general'
    return tsla.Solve(A_structure)(arr.astype('float32'), I).astype('float64')


def inv(a):
    if is_tensor(a): return thinv(a)
    return ar.inv(a)


def eye(n, m=None, k=0, dtype=None):
    if is_tensor(n): return T.eye(n, m, k, dtype)
    return np.eye(n, m, k, dtype)


def fill_diagonal(a, val):
    if is_tensor(a): return tensor_fill_diagonal(a, val)
    return np.fill_diagonal(a, val)


def dotT(weight, matrix):
    """Transpose matrix multiplication. 
    Transpose matrix and result as well. Purpose is to allow for vectors within each row that are weighed by the weight matrix.
    :param weight: sparse or dense weight m x n matrix, where m is number of features of output and n is number of 
    features for input.
    :param matrix: input n x p matrix, where n is number of features for input and p is number of input vectors.
    """
    assert weight is not None and matrix is not None
    return T.transpose(dot(weight, T.transpose(matrix)))


def zeros_like(sparse_matrix, name=None):
    """Create a sparse matrix with indices like another with values that are zero."""
    # get scipy sparse csc matrix
    try:
        sparse_matrix = sparse_matrix.get_value()
    except AttributeError:
        pass
    matrix = sparse_matrix.copy()
    matrix.data[:] = 0
    return th.shared(matrix) if name is None else th.shared(matrix, name)


def MSE(y, target):
    """
    Mean squared error used typically as cost function for theano network.
    :param y: prediction_function matrix
    :param target: target matrix
    :return: a symbolic theano scalar variable that is the MSE cost
    """
    return T.mean((y - target) ** 2) / 2


def set_indices(tensor, indices, value):
    """
    This is supposedly still differentiable using magic.
    Set fields in tensor at given indices to the given value
    :param tensor: a theano tensor to change.
    :param indices: boolean indexing tensor (mask) to choose the positions to change.
    :param value: the value to insert.
    :return: the changed tensor, where the positions are set to the given value.
    """
    return T.set_subtensor(tensor[indices], value)


def lasso_cost(weights, lasso):
    """
    Lasso cost for sparse or dense weight matrices
    :param weights: sparse or dense matrix shared variable or list of them
    :param lasso: lasso coefficient weighing the cost
    :return: the cost, theano scalar variable
    """
    if not isinstance(weights, list): weights = [weights]
    sums = []
    for weight in weights:
        if is_sparse(weight): weight = sp.csm_data(weight)
        sums.append(T.mean(T.abs_(weight)))
    return sum(sums) * lasso


def is_tensor(a):
    return hasattr(a, "eval")






def training_function(inputs, weights, prediction, target, gradient_descent, lasso):
    """
    Define a theano training function for a defined network with given theano variables.
    :param inputs: list of theano variables that are input into a defined network
    :param weights: list of theano variables that are weight matrices in a defined network
    :param prediction: theano variable to predict
    :param target: theano variable to compare the prediction to
    :param gradient_descent: string name of the gradient descent to use, e.g. "adam" or "sgd"
    :param lasso: float lasso coefficient
    :return: theano training_function function
    """
    cost = mse = MSE(prediction, target)
    if lasso: cost += lasso_cost(weights, lasso)
    updates = gd.updates[gradient_descent](cost=cost, params=weights)
    return th.function(inputs=inputs + [target], outputs=[cost, mse], updates=updates, allow_input_downcast=True)


def prediction_function(inputs, prediction):
    """
    Get a theano prediction function
    :param inputs: list of theano variables that are the inputs of the network
    :param prediction: theano variable(s) to predict
    :return: theano function for prediction
    """
    return th.function(inputs=inputs, outputs=prediction, allow_input_downcast=True)


def train(train_function, inputs, target, start_epoch=0, epochs=100, n_prints=100, batch_size=None, axis=0):
    """
    Perform the train loop given a train function
    :param train_function: theano training function
    :param inputs: list given as inputs to the network, e.g. list of arrays
    :param target: the target values, e.g. an array
    :param start_epoch: the starting epoch to count from
    :param epochs: number of times the entire dataset is reused
    :param n_prints: number of times to print status
    :param batch_size: number of datapoints used at a time. 
    If not provided, it will increase from 1 to max through train.
    :param axis: argument for the yield_batch function
    :return: None
    """
    # check inputs
    assert isinstance(inputs, list)

    # check error at epoch zero
    print("Std before training = %.6f" % math.sqrt(train_function(*inputs, target)[-1]))

    costs, mses, outputs = [], [], []
    print_epochs = np.linspace(start_epoch, start_epoch + epochs, n_prints, endpoint=False).round()
    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):
        progress = ma.lerp_inverse(start_epoch, start_epoch + epochs, epoch)
        output = {
            "epoch": epoch,
            "batch": ma.lerp_int(1, len(target), progress) if batch_size is None else batch_size
        }

        # train on each batch
        for batch in neu.yield_batch(*inputs, target, batch_size=output["batch"], axis=axis):
            cost, mse = train_function(*batch)
            costs.append(cost)
            mses.append(mse)

        # time to print status
        if output["epoch"] in print_epochs:
            output["cost"] = np.sqrt(costs).mean()
            output["std"] = np.sqrt(mses).mean()
            output["time"] = time.time() - start_time
            outputs.append(output)
            costs, mses = [], []
            neu.print_train(output)

    return pd.DataFrame(outputs)


def train_std(train_functions, inputs, target, weights, start_epoch=0, epochs=100, n_prints=100, batch_size=None):
    """
    Perform the train loop given a train function and all the relevant values
    :param train_functions: theano training functions
    :param inputs: values given as inputs to the network, e.g. list of arrays
    :param target: the target values, e.g. an array
    :param weights: list of list of shared theano variables holding the weights
    :param start_epoch: the starting epoch to count from
    :param epochs: number of times the entire dataset is reused
    :param n_prints: number of times to print status
    :param batch_size: number of datapoints used at a time. 
    If not provided, it will increase from 1 to max through train.
    :return: None
    """
    reps = len(train_functions)
    # check error at epoch zero using the last one as a random choice
    print("Std before training = %.6f" % math.sqrt(train_functions[-1](*inputs, target)[-1]))

    costs, mses, outputs = [[] for _ in range(reps)], [[] for _ in range(reps)], []
    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):
        progress = ma.lerp_inverse(start_epoch, epochs, epoch)
        output = {"batch": ma.lerp_int(1, len(target), progress) if batch_size is None else batch_size}

        # train on each batch on each repetition
        batch_generators = [neu.yield_batch(*inputs, target, batch_size=output["batch"]) for _ in range(reps)]
        for batches in zip(*batch_generators):
            for i, batch in enumerate(batches):
                cost, mse = train_functions[i](*batch)
                costs[i].append(cost)
                mses[i].append(mse)

        # time to print status
        if epoch % (epochs / n_prints) == 0:
            output = {
                **output,
                "epoch": epoch,
                "cost": np.sqrt(costs).mean(),
                "std": np.sqrt(mses).mean(),
                "time": time.time() - start_time,
                "rep std 1": std_across([w[0] for w in weights]),
                "rep std 2": std_across([w[1] for w in weights]),
                "rep std 3": std_across([w[2] for w in weights])
            }

            outputs.append(output)
            costs, mses = [[] for _ in range(reps)], [[] for _ in range(reps)]
            neu.print_train(output)

    return pd.DataFrame(outputs)





def l1(arr, axis=None):
    """Return the L1 norm of a tensor.
    Parameters
    ----------
    arr : Theano variable.
        The variable to calculate the norm of.
    axis : integer, optional [default: None]
        The sum will be performed along this axis. This makes it possible to
        calculate the norm of many tensors in parallel, given they are organized
        along some axis. If not given, the norm will be computed for the whole
        tensor.
    Returns
    -------
    res : Theano variable.
        If ``axis`` is ``None``, this will be a scalar. Otherwise it will be
        a tensor with one dimension less, where the missing dimension
        corresponds to ``axis``.
    Examples
    --------
    >>> v = T.vector()
    >>> this_norm = l1(v)
    >>> m = T.matrix()
    >>> this_norm = l1(m, axis=1)
    >>> m = T.matrix()
    >>> this_norm = l1(m)
    """
    return abs(arr).sum(axis=axis)


def soft_l1(inpt, eps=1e-8, axis=None):
    """Return a "soft" L1 norm of a tensor.
    The term "soft" is used because we are using :math:`\sqrt{x^2 + \epsilon}`
    in favor of :math:`|x|` which is not smooth at :math:`x=0`.
    Parameters
    ----------
    arr : Theano variable.
        The variable to calculate the norm of.
    eps : float, optional [default: 1e-8]
        Small offset to make the function more smooth.
    axis : integer, optional [default: None]
        The sum will be performed along this axis. This makes it possible to
        calculate the norm of many tensors in parallel, given they are organized
        along some axis. If not given, the norm will be computed for the whole
        tensor.
    Returns
    -------
    res : Theano variable.
        If ``axis`` is ``None``, this will be a scalar. Otherwise it will be
        a tensor with one dimension less, where the missing dimension
        corresponds to ``axis``.
    Examples
    --------
    >>> v = T.vector()
    >>> this_norm = soft_l1(v)
    >>> m = T.matrix()
    >>> this_norm = soft_l1(m, axis=1)
    >>> m = T.matrix()
    >>> this_norm = soft_l1(m)
    """
    return T.sqrt(inpt ** 2 + eps).sum(axis=axis)


def l2(arr, axis=None):
    """Return the L2 norm of a tensor.
    Parameters
    ----------
    arr : Theano variable.
        The variable to calculate the norm of.
    axis : integer, optional [default: None]
        The sum will be performed along this axis. This makes it possible to
        calculate the norm of many tensors in parallel, given they are organized
        along some axis. If not given, the norm will be computed for the whole
        tensor.
    Returns
    -------
    res : Theano variable.
        If ``axis`` is ``None``, this will be a scalar. Otherwise it will be
        a tensor with one dimension less, where the missing dimension
        corresponds to ``axis``.
    Examples
    --------
    >>> v = T.vector()
    >>> this_norm = l2(v)
    >>> m = T.matrix()
    >>> this_norm = l2(m, axis=1)
    >>> m = T.matrix()
    >>> this_norm = l2(m)
    """
    # not sure why we have to use the root
    # return T.sqrt((arr ** 2).sum(axis=axis) + 1e-8)
    return (arr ** 2).sum(axis=axis)


def lp(inpt, p, axis=None):
    """Return the Lp norm of a tensor.
    Parameters
    ----------
    arr : Theano variable.
        The variable to calculate the norm of.
    p : Theano variable or float.
        Order of the norm.
    axis : integer, optional [default: None]
        The sum will be performed along this axis. This makes it possible to
        calculate the norm of many tensors in parallel, given they are organized
        along some axis. If not given, the norm will be computed for the whole
        tensor.
    Returns
    -------
    res : Theano variable.
        If ``axis`` is ``None``, this will be a scalar. Otherwise it will be
        a tensor with one dimension less, where the missing dimension
        corresponds to ``axis``.
    Examples
    --------
    >>> v = T.vector()
    >>> this_norm = lp(v, .5)
    >>> m = T.matrix()
    >>> this_norm = lp(m, 3, axis=1)
    >>> m = T.matrix()
    >>> this_norm = lp(m, 4)
    """
    return ((inpt ** p).sum(axis=axis)) ** (1. / p)


def relu(v, offset=0):
    return T.nnet.relu(v - offset)





