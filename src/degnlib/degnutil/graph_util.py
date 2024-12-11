#!/usr/bin/env python3
import pandas as pd
from degnutil.pandas_util import concat_pandas


def to_edges(array, threshold=0.):
    """
    Convert an array to DataFrame of edges, using a cutoff to remove weak connections. 
    The values in the array are put in a column named "weight"
    :param array: pandas array, column names are from nodes, index names are target nodes
    :param threshold: float threshold. keep edge with weight x if abs(x) > threshold
    :return: pandas DataFrame
    """
    return pd.DataFrame(list(yield_edges(array, threshold=threshold)), columns=["source", "target", "weight"])


def yield_edges(array, threshold=0.):
    """
    Yield edges from array. Optional threshold can be used to ignore weak connections.
    :param array: pandas array, column names are from nodes, index names are target nodes
    :param threshold: float threshold. yield edge with weight x if abs(x) > threshold
    :return: generator [source, target, weight]
    """
    for source in array:
        for target in array.index:
            weight = array[source][target]
            if abs(weight) > threshold:
                yield [source, target, weight]


def arrays_to_edges(arrays, threshold=0., group=""):
    """
    Use multiple arrays where edges are taken from all if they are stronger than a threshold.
    Optionally values from each group get a descriptor added to a new grouping column to tell them apart.
    :param arrays: collection of arrays, use dict so dict keys can be used as the descriptor of the array.
    :param threshold: float threshold to filter edge strength.
    :param group: string name to use for a new column to tell original arrays apart.
    :return: pandas DataFrame
    """
    try:
        arrays = {k: to_edges(a, threshold=threshold) for k, a in arrays.items()}
    except AttributeError:
        arrays = [to_edges(a, threshold=threshold) for a in arrays]
    return concat_pandas(arrays, group)


def to_nodes(names, **properties):
    """
    Convert a collection of names to pandas DataFrame with columns "name", and keys from properties argument.
    :param names: list or other collection
    :param properties: keyword properties, e.g. type="pk"
    :return: pandas DataFrame
    """
    nodes = pd.DataFrame(names, columns=["name"])
    for k, v in properties.items(): nodes[k] = v
    return nodes


def names_to_nodes(names):
    """
    Use multiple name lists.
    Optionally values from each group get a descriptor added to a new grouping column to tell them apart.
    :param names: collection of name collections, e.g. dict of strings.
    use dict so dict keys can be used as the descriptor of the list elements.
    :return: pandas DataFrame
    """
    return concat_pandas({k: to_nodes(n) for k, n in names.items()}, group="type")


