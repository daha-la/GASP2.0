#!/usr/bin/env python3
import numpy as np
from utilities import design as de
from degnutil import string_util as st, math_util as ma

"""
This script is for neural network methods, e.g. code for batch sizes and train procedures.
"""


def yield_batch(*args, batch_size=1, axis=0):
    """
    Yield a batch one at a time from array.
    :param args: list of arrays. 
    :param batch_size: number of datapoints given each yield
    :param axis: the axis to yield batches along. This shape[axis] should be the same for all arrays given in args.
    :return: a generator for batches
    """
    n = args[0].shape[axis]
    for arg in args:
        assert arg.shape[axis] == n, "There is different number of datapoints among the array collections."
    
    indexes = np.arange(n)
    np.random.shuffle(indexes)
    for i in range(0, n, batch_size):
        index = indexes[i:i+batch_size]
        batches = [arg.take(index, axis=axis) for arg in args]
        yield tuple(batches)


def print_train(outputs, color=True):
    """
    Format and print one line of info for training
    :param outputs: dict of values to print
    :param color: bool indicating if we will color and format text
    :return: 
    """
    # maximum std that gives the upper bound of coloring.
    color_max_std = 0.2
    outputs = outputs.copy()

    values = outputs.copy()
    keys = {k: k + "=" for k in outputs.keys()}
    give_color = {k: False for k in outputs.keys()}
    
    values["time"] = st.seconds_to_string(outputs["time"]).rjust(16, ' ')

    for k, v in values.items():
        try: v = "{:4d}".format(v)
        except ValueError:
            try: v = "{:9.6f}".format(v)
            except ValueError: pass
            # if it is float we want to color it
            else: give_color[k] = True
        values[k] = v

    if color:
        keys = {k: de.dim(v) for k, v in keys.items()}
        values["epoch"] = de.bright(values["epoch"])
        values["std"] = de.bright(values["std"])

    messages = {k: keys[k] + values[k] for k in values.keys()}

    if color:
        for k in messages.keys():
            if not give_color[k]: continue
            # a number from 0 to 1 where 1 means min error, and 0 means max error
            try: performance = 1 - min(1, outputs[k] / color_max_std)
            except TypeError: continue
            # color from red to green
            messages[k] = de.color_back(messages[k], ma.lerp(de.red_hue, de.green_hue, performance), performance, performance)

    print("\t".join([messages[k] for k in sorted(messages.keys())]))

