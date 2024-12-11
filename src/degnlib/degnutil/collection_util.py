#!/usr/bin/env python3

"""
This script is for utilities related to collections, 
e.g. lists, sets, dicts.
"""

def dict_upper(dictionary):
    """
    Convert a dictionary to uppercase keys and values.
    :param dictionary: dict to convert
    :return: dict converted
    """
    return {k.upper():v.upper() for k, v in dictionary.items()}


def set_upper(s):
    return set(v.upper() for v in s)


def unjag_list(l):
    """
    Unjag a list
    :param l: jagged list
    :return: unjagged list
    """
    out = []
    for v in l:
        if isinstance(v, list):
            out += unjag_list(v)
        else: out.append(v)
    return out


def is_iterable(c):
    try: iter(c)
    except TypeError: return False
    return True


