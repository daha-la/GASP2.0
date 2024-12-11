#!/usr/bin/env python3
import time
import math
import numpy as np
import pandas as pd
import torch as tc
import torch.nn.functional as F
from degnlib.degnutil.neural_network import print_train
from degnlib.degnutil import array_util as ar, neural_network as neu

"""
This script is for utilities related to pytorch.
"""

def tensor(arr, requires_grad=False):
    """
    Return float32 tensor.
    :param arr: 
    :param requires_grad: If gradients should be described for the tensor so we can train it
    :return: 
    """
    try: return tc.tensor(ar.fl(arr), requires_grad=requires_grad)
    except (ValueError, TypeError):
        # assume pandas
        return tc.tensor(ar.fl(arr.values), requires_grad=requires_grad)


def toarray(tensor):
    """
    Convert a pytorch tensor to a numpy array
    :param tensor: torch tensor
    :return: dense numpy array
    """
    return tensor.detach().numpy()


def inv(a):
    """
    Inverse a matrix.
    :param a: Either pytorch tensor or numpy array.
    :return: inverse of matrix "a" of same type as given.
    """
    try: return a.inverse()
    except AttributeError: return ar.inv(a)


def ascol(tensor1d):
    """
    Reshape a 1d tensor to a 2D column tensor
    :param tensor1d: 1D tensor
    :return: 2D column tensor
    """
    return tensor1d.unsqueeze(1)


def col(tensor, j):
    """
    Get the jth column in an tensor where the result is a 2D column tensor.
    :param tensor: 2D tensor
    :param j: int index along axis 1
    :return: 2D column tensor
    """
    return tensor[:,j:j+1]


def super_mm(As, B):
    """
    Pytorch helper function
    Do matrix multiplication (dot) as A . B but for each column in B a different A is used. 
    :param As: list of As, assumed to have length of B shape 1 (width)
    :param B: 
    :return:
    """
    tensors = [As[i] @ col(B, i) for i in range(len(As))]
    return tc.cat(tuple(tensors), 1)


def fill_diagonal(tensor, val):
    """
    Fill diagonal of tensor with val. 
    :param tensor: 
    :param val: 
    :return: 
    """
    n, m = tensor.shape
    I = tc.eye(n, m)
    out = tensor * (1 - I)
    if not val: return out
    return out + val * I


def parse_optim(name):
    for n in [name, name.upper(), name.capitalize()]:
        try: return getattr(tc.optim, n)
        except AttributeError: pass
    raise AttributeError("No such optim func found.")


def train(model, inputs, optimizer, epochs, n_prints=20, start_epoch=0, extra=None):
    """
    Run training loop with no batch.
    :param model: Function to call given data and trainable weight(s) that returns loss and mse.
    :param inputs: list of inputs to give the model.
    :param optimizer: pytorch optimizer from torch.optim
    :param epochs: int number of epochs to train for
    :param n_prints: number of times to print a line for status.
    :param start_epoch: epoch to count relative from.
    :param extra: function returning dict for extra outputs per print. The function is evaluated for each print.
    :return: table with outputs from the training
    """
    
    # check error at epoch zero
    print("Std before training = {:.6f}".format(math.sqrt(model(*inputs)[-1])))

    costs, mses, outputs = [], [], []
    print_epochs = np.linspace(start_epoch, start_epoch + epochs, n_prints, endpoint=False).round()
    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):
        
        optimizer.zero_grad()
        cost, mse = model(*inputs)
        cost.backward()
        optimizer.step()
        
        costs.append(cost.item())
        mses.append(mse.item())

        # time to print status
        if epoch in print_epochs:
            output = {
                "epoch": epoch,
                "cost": np.sqrt(costs).mean(),
                "std": np.sqrt(mses).mean(),
                "time": time.time() - start_time,
            }
            if extra is not None: output = {**output, **extra()}
            
            print_train(output)
            outputs.append(output)
            costs, mses = [], []
    
    return pd.DataFrame(outputs)


def train_batch(model, inputs, optimizer, epochs, batch_size, n_prints=20, start_epoch=0, extra=None):
    """
    Run training loop with no batch.
    :param model: Function to call given data and trainable weight(s) that returns loss and mse.
    :param inputs: list of inputs to give the model that should be batched.
    :param optimizer: pytorch optimizer from torch.optim
    :param epochs: int number of epochs to train for
    :param batch_size: int number of datapoints per gradient descent step
    :param n_prints: number of times to print a line for status.
    :param start_epoch: epoch to count relative from.
    :param extra: function returning dict for extra outputs per print. The function is evaluated for each print.
    :return: table with outputs from the training
    """

    # check error at epoch zero
    print("Std before training = {:.6f}".format(math.sqrt(model(*inputs)[-1])))

    costs, mses, outputs = [], [], []
    print_epochs = np.linspace(start_epoch, start_epoch + epochs, n_prints, endpoint=False).round()
    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):

        # train on each batch
        for batch in neu.yield_batch(*inputs, batch_size=batch_size, axis=1):
            optimizer.zero_grad()
            # mse= mean squared error, so it makes the most sense to have it be the mean over all the squared errors.
            cost, mse = model(*batch)
            cost.backward()
            optimizer.step()
            costs.append(cost.item())
            mses.append(mse.item())

        # time to print status
        if epoch in print_epochs:
            output = {
                "epoch": epoch,
                "cost": np.mean(costs),
                "std": np.sqrt(mses).mean(), 
                "time": time.time() - start_time,
            }
            if extra is not None: output = {**output, **extra()}
            
            print_train(output)
            outputs.append(output)
            costs, mses = [], []

    return pd.DataFrame(outputs)


def relu(input):
    return F.relu(input)




