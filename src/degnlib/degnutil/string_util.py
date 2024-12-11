#!/usr/bin/env python3
from datetime import timedelta
import numpy as np
import re

"""
This script is for utilities related to strings and text.
"""

def has_match(pattern, string):
    """
    :param pattern: regex
    :param string:
    :return: bool, was there a match
    """
    return len(re.findall(pattern, string)) > 0


def parse_floatint(v):
    """
    Parse a string known to contain either float or int.
    :param v: string representation of an int or float
    :return: int or float
    """
    try: return int(v)
    except TypeError: return float(v)


def parse_number(v):
    """
    Given a string that is a number, convert it to the correct number type.
    :param v: a single number with string type (or only item in collection).
    :return: an int or a float

    Examples:
    "4.0" -> 4.0
    "4" -> 4
    np.asarray([[3]]) -> 3
    """
    try: return parse_value(v.item())
    except (AttributeError, ValueError):
        # we need to make sure the v is str type when trying to convert to integer, 
        # since it would be successfully converted if v was actually v float.
        try: return int(str(v))
        except (ValueError, TypeError):
            return float(v)
            

def parse_value(v):
    """
    Given a string that may be a number, convert it to the correct number type.
    :param v: a string maybe with a single number (or only item in collection).
    :return: an int, a float or string

    Examples:
    "4.0" -> 4.0
    "4" -> 4
    np.asarray([[3]]) -> 3
    "hi" -> "hi"
    """
    try: return parse_number(v)
    except (ValueError, TypeError): return v


def parse_values(vs):
    return [parse_value(v) for v in vs]


def parse(v):
    """
    Parse some value that might be a lot of things.
    
    Examples:
    "True" -> True
    "4.0" -> 4.0
    "4" -> 4
    np.asarray([[3]]) -> 3
    "hi" -> "hi"
    :param v: 
    :return: 
    """
    if v == "True": return True
    if v == "False": return False
    return parse_value(v)


def seconds_to_string(seconds):
    """
    Convert a number of seconds to a formatted string representation
    :param seconds: int
    :return: str
    """
    return str(timedelta(seconds=round(seconds)))


def time_to_string(**kwargs):
    """
    Get time formatted from kwargs looking for a seconds, minutes and hours entry.
    It is flexible enough that there can be other keys present and all keys are optional.
    :param kwargs: keywords given maybe including time keywords
    :return: string describing delta time
    """
    time = str(timedelta(**{k: kwargs.get(k, 0) for k in ["seconds", "minutes", "hours"]}))
    return time.replace(" days, ", ":")
    

def join(a, sep):
    """
    Like builtin join, but also allows numbers
    :param a: list of str, int, float
    :param sep: 
    :return: str
    """
    return sep.join([str(v) for v in a])


def is_empty(s):
    """
    Test if string is empty, or is describing an empty string itself.
    :param s: string
    :return: bool
    """
    return s in ['', '""', "''"]


def binary(num, width=None):
    """
    Function made to handle the case of representing an int as binary 
    but where a choice of zero length is actually an empty string.
    This is not always what you want which is not intuitive.
    :param num: 
    :param width: 
    :return: 
    """
    if width == 0: return ''
    return np.binary_repr(num, width)


def subslice(string, start, end, origin=0, pad=0):
    """
    Get a range for a substring, which lacks python's negative indexes and index step size.
    Currently, if end < start and pad is big enough it may return a non-empty string.
    :param string: the string to substring.
    :param start: inclusive first position
    :param end: inclusive last position
    :param origin: is it zero indexed or one indexed numbering in "start" and "end"? Default=0
    :param pad: add this many letters at either end. Can be a negative amount.
    :return: slice of simple string which is contained somewhere in "string"
    """
    start = max(0, start - origin - pad)
    stop = min(len(string), max(start, end - origin + pad + 1))  # +1 since the stop index goes 1 further than an end index
    return slice(start, stop)


def substring(string, start, end, origin=0, pad=0):
    """
    Get a substring, which lacks python's negative indexes and index step size.
    Currently, if end < start and pad is big enough it may return a non-empty string.
    :param string: the string to substring.
    :param start: inclusive first position
    :param end: inclusive last position
    :param origin: is it zero indexed or one indexed numbering in "start" and "end"? Default=0
    :param pad: add this many letters at either end. Can be a negative amount.
    :return: simple string which is contained somewhere in "string"
    """
    return string[subslice(string, start, end, origin, pad)]


def count_uppercase(string):
    """
    
    :param string: str, Seq, or SeqRecord 
    :return: int
    """
    return sum(np.char.asarray(list(string)).isupper())


def delete_lowercase(string):
    """
    Lowercase letters deleted from sequence.
    :param string: str, Seq, or SeqRecord
    :return: str
    """
    arr = np.char.asarray(list(string))
    return ''.join(arr[~arr.islower()])


def re_range(string):
    """
    Match for range, i.e. two numbers separated by 1-3 non-alphanumeric character.
    Could be restricted to only identical non-alphanumeric but maybe it's fine.
    :param string: str, e.g. "43-234" or "223...121"
    :return: (int start, int end) or None
    """
    if not re.match('\d+\W{1,3}\d+', string): return None
    return int(re.match("\d+", string)[0][0]), int(re.findall("\d+$", string)[0][0])


