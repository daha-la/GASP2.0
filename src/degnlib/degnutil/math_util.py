#!/usr/bin/env python3
import time
import numpy as np

"""
This script is for utilities related to math.
"""


def sqrt(x):
    """
    32 bit sqrt
    :param x: scalar or array
    :return: sqrt of x
    """
    return np.sqrt(x, dtype=np.float32)


def clip(value, minimum, maximum):
    if value < minimum: return minimum
    if value > maximum: return maximum
    return value


def lerp(start, end, progress):
    """
    Interpolation between a start and an end value using a factor indicating progress.
    :param start: value to start with
    :param end: value to end with
    :param progress: number between 0 and 1 indicating how far we are between the values
    :return: 
    """
    if progress <= 0: return start
    if progress >= 1: return end
    return start * (1 - progress) + end * progress


def lerp_int(start, end, progress):
    """
    whole number interpolation that provides the same amount of time at each value between start and end 
    if we progress linearly between them.
    :param start: int inclusive
    :param end: int inclusive
    :param progress: float
    :return: int
    """
    return clip(round(lerp(start-.5, end+.5, progress)), start, end)


def lerp_log(start, end, progress):
    """Interpolation between a start and an end value using a factor indicating progress. 
    The interpolation is not linear, it is logarithmic."""
    if progress <= 0: return start
    if progress >= 1: return end
    return np.exp(progress * (np.log(end) - np.log(start)) + np.log(start))


def lerp_time(start_value, end_value, start_time, duration):
    """Lerp between a start and end value logarithmically using time progression."""
    return lerp(start_value, end_value, lerp_inverse(start_time, start_time + duration, time.time()))


def lerp_log_time(start_value, end_value, start_time, duration):
    """Lerp between a start and end value logarithmically using time progression."""
    return lerp_log(start_value, end_value, lerp_inverse(start_time, start_time + duration, time.time()))


def lerp_inverse(start, end, value):
    """:return a value (ideally) from zero to one indicating how far a value has progressed between a start and an end."""
    return (value - start) / (end - start)
