#!/usr/bin/env python3
import theano
from theano import sparse as sp
import theano.tensor as T
import numpy as np
from degnlib.degnutil import theano_util as thea

"""
Different update functions that can be used with theano.
"""


def sgd(cost, params, learning_rate=0.001):
    """
    Basic stochastic gradient descent for sparse weight matrices.
    :param cost: A theano scalar variable to minimize
    :param params: A list of theano variables that store the sparse weights.
    :param learning_rate: A number to indicate how fast to perform gradient descent.
    :return: A list of tuples where the first element is the variable to update and the second element is the update.
    """
    if not isinstance(params, list): params = [params]
    updates = []
    grads = theano.grad(cost, params)

    for param, grad in zip(params, grads):
        updates.append((param, param - grad * learning_rate))
    return updates


def adam(cost, params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    
    Modified for possibility of sparse param weight matrices by converting to dense and then back.
    The conversion happens after theano.grad is run so I hope the performance is still good.
    """
    # we might have a single weight matrix or multiple in a list all a part of a model in different places.
    if not isinstance(params, list): params = [params]
    updates = []
    grads = theano.grad(cost, params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    # Decay the first moment running average coefficient
    b1_t = b1*gamma**(t-1)

    for param, grad in zip(params, grads):
        sparse = thea.is_sparse(param)
        if sparse:
            theta_previous = sp.dense_from_sparse(param)
            g = sp.dense_from_sparse(grad)
        else:
            theta_previous = param
            g = grad
        
        
        m_previous = theano.shared(np.zeros(param.get_value().shape, dtype='float32'))
        v_previous = theano.shared(np.zeros(param.get_value().shape, dtype='float32'))
        # Update biased first moment estimate
        m = b1_t*m_previous + (1-b1_t) * g
        # Update biased second raw moment estimate
        v = b2*v_previous + (1-b2)*g**2
        # Compute bias-corrected first moment estimate
        m_hat = m / (1-b1**t)
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1-b2**t)
        # Update parameters
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e)

        updates.append((m_previous, m.astype('float32')))
        updates.append((v_previous, v.astype('float32')))
        if sparse: updates.append((param, sp.csc_from_dense(theta)))
        else: updates.append((param, theta.astype('float32')))
    updates.append((t, t + 1.))
    return updates


updates = {'sgd':sgd, 'adam':adam}

