#!/usr/bin/env python3
import numpy as np

palette = np.asarray(['darkorange', 'navy', 'forestgreen', 'maroon', 'purple', 'red', 'gold', 'blue', 'lime', 'deepskyblue', 'black', 'gray', 'cornflowerblue', 'seagreen', 'tomato', 'olive'])

def unique_legends(legends):
    """
    Given a list of legends with replicates, 
    get the list of legends and colors needed to plot the curves so that curves with the same legend get the same color.
    Duplicate legend entries are set to None so that only one legend entry will be shown for each group.
    :param legends: list of legend strings
    :return: (list of legend strings also with None values, list of colors with duplicates)
    """
    unique, unique_index, unique_inverse = np.unique(legends, return_index=True, return_inverse=True)
    # have the same color for each unique legend
    colors = palette[unique_inverse]
    # have None for legend unless it is first unique instance of a legend string
    legends = np.asarray([None for _ in legends])
    legends[unique_index] = unique
    return legends, colors

