#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.File import as_handle
import re
from degnutil import string_util as st, input_output as io
from degnutil.input_output import log

"""
This script is for reading and writing to files. 
It is not meant to be run on its own, but rather support other scripts.
"""

def read_array(handleish, sep=None):
    """
    General function that reads arrays whether or not they have a certain delimiter, an index and header.
    NOTE: If the array has accidental spaces at the end of some lines it will fail to be read as numpy and will be read as pandas 
    where an extra column will be made with NaNs.
    :param handleish: string, Path, or sys.stdin
    :param sep: separator character
    :return: pandas array if there are named columns/rows or numpy array otherwise
    """
    # if stdin, move to memory
    path = io.save_if_stream(handleish)
    # simple int matrix with whitespace delimiter
    try: return np.loadtxt(path, dtype=int)
    except ValueError: pass
    # simple int matrix with detected delimiter
    if sep is None: sep = detect_delimiter(path)
    try: return np.loadtxt(path, dtype=int, delimiter=sep)
    except ValueError: pass
    # simple float matrix with detected delimiter
    try: return np.loadtxt(path, delimiter=sep)
    except ValueError: pass
    # try bool
    try: return read_bool(path, delimiter=sep)
    except ValueError: pass
    # fall back on pandas
    has_index = table_has_index(path, sep)
    if has_index: return pd.read_csv(path, index_col=0, sep=sep)
    return pd.read_csv(path, sep=sep)


def read_bool(path, delimiter=" "):
    """
    Read a boolean array where each element is written as string, i.e. "True", "False"
    :param path: path to array
    :param delimiter: 
    :return: numpy array
    """
    arr_str = np.loadtxt(path, dtype=str, delimiter=delimiter)
    out = arr_str == "True"
    # make sure that all values are either "True" or "False"
    if not np.all(out | (arr_str == "False")):
        raise ValueError("File contains non-bool values")
    return out


def write_array(array, path, sep=' '):
    """
    Write array to path.
    :param array: 
    :param path: 
    :param sep: delimiter character
    :return: None
    """
    # pandas write function
    try:
        # if index is not named and is the Int64Index type it should not be written to file by default
        index = array.index.name or not isinstance(array.index, pd.Int64Index)
        array.to_csv(path, sep=sep, index=index)
    except AttributeError:
        # convert to use pandas write function since it formats better than numpy's
        pd.DataFrame(array).to_csv(path, sep=sep, header=False, index=False)


def detect_delimiter(path):
    """
    Detect the delimiter in a file. 
    :param path: string or Path
    :return: string delimiter
    """
    seps = ['\t', ',', ' ', ':', ';']
    counts = []
    
    for line in Path(path).open():
        counts.append([line.count(sep) for sep in seps])
    
    # now we have a table of counts of each possible separator
    counts = np.asarray(counts)
    means = counts.mean(axis=0)
    stds = counts.std(axis=0)
    
    # order from most to least likely.
    # most important is whether there is on average more than one of sep at any line
    # then it is important whether it varies more than 1
    # then sorted by having least deviation
    # and finally by having the largest frequency 
    order = np.lexsort((-means, stds, stds >= 1, means <= 1))
    # the highest sorted sep character is chosen
    return np.asarray(seps)[order[0]].item()


def table_has_index(path, sep):
    """
    Detect whether a file has an index column as the first column of the file.
    :param path: string or Path
    :param sep: value separator
    :return: bool
    """
    handle = Path(path).open()
    line = next(handle).split(sep)
    # check if first cell of file is empty, it is empty if there is an index
    if not st.is_empty(line[0]): return False
    for line in handle:
        # check if first cell of other lines are empty, 
        # it would not be a complete index if they are
        if not line[0]: return False
    return True


class Edit:
    """
    Edit a file by creating a backup with extension .bak that is read while writing to a file with the original filename.
    :param path: file to edit
    :return: class object that can be used with the "with" keyword to return a tuple of two file handles, 
    first to be read, second to be written to.

    Example:
    with Edit(path) as (infile, outfile):
        for line in infile:
            outfile.write(line)
    """
    def __init__(self, path):
        path = str(path)
        
        # move to backup
        try: os.rename(path, path + ".bak")
        except FileNotFoundError:
            if not os.path.isfile(path + ".bak"):
                raise FileNotFoundError("Cannot find file: " + path)
            # otherwise there is a backup present so we use that for reading still
        self.infile = open(path + ".bak", 'r')
        self.outfile = open(path, 'w')
    
    def __enter__(self):
        """
        This function decides what is put in the variable following "as" keyword
        :return: tuple of infile, outfile
        """
        return self.infile.__enter__(), self.outfile.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.infile.__exit__(exc_type, exc_val, exc_tb)
        self.outfile.__exit__(exc_type, exc_val, exc_tb)


def read_class(path):
    import pickle
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def write_class(class_instance, path):
    import pickle
    with open(path, 'wb') as fh:
        pickle.dump(class_instance, fh)


def read_vector(handleish, delimiter=None, skip0=False):
    """
    Read file where it is known to contain a single vector of values, each possibly named.
    :param handleish: 
    :param delimiter: default=None meaning whitespace
    :param skip0: skip first line?
    :return: list or dict
    """
    with as_handle(handleish) as infile:
        if skip0: next(infile)
        array = np.asarray([line.strip().split(delimiter) for line in infile])
        if array.shape[0] == 1: return st.parse_values(array[0,:])
        if array.shape[1] == 1: return st.parse_values(array[:,0])
        # must be named otherwise
        if array.shape[0] == 2: return dict(zip(st.parse_values(array[0,:]), st.parse_values(array[1,:])))
        if array.shape[1] == 2: return dict(zip(st.parse_values(array[:,0]), st.parse_values(array[:,1])))
    
    raise IOError("File content not recognized as vector.")


def yield_field(strings, key, fieldsep=" ", kvsep="="):
    """
    Yield field value given a key that is expected to be found once in each string.
    :param strings: 
    :param key: 
    :param fieldsep: 
    :param kvsep: 
    :return: yields the value (str) if found for a string, otherwise None
    """
    if isinstance(strings, str): strings = [strings]
    for string in strings:
        val = None
        for field in string.split(fieldsep):
            kv = field.split(kvsep)
            if len(kv) == 2 and kv[0] == key:
                assert val is None, "Field found multiple times."
                val = kv[1]
        yield val


def read_text_integers(handleish, sep=None, header=False):
    """
    Read and separate text and integers, where integers can be found in the form of ranges, e.g. "4-10"
    Assert that there is consistent amounts of text and number cells found in each line.
    :param handleish: 
    :param sep: 
    :param header: 
    :return: [([text,...],[number,...]),...]
    """
    texts, numbers = _read_text_integers(handleish, sep, header)
    n_text, n_numbers = len(texts[0]), len(numbers[0])
    for text, number in zip(texts, numbers):
        assert len(text)   == n_text,    "Inconsistent amount of text cells found among rows"
        assert len(number) == n_numbers, "Inconsistent amount of number cells found among rows"
    return list(zip(texts, numbers))


def read_id_integers(handleish, sep=None, header=False):
    """
    Same as read_text_integers except the first text column is the only text column returned.
    """
    texts, numbers = _read_text_integers(handleish, sep, header)
    for i, t in enumerate(texts):
        try: texts[i] = t[0]
        except IndexError: texts[i] = ""
    
    return list(zip(texts, numbers))


def read_ranges(handleish, sep=None, header=False):
    """
    Same as read_id_integers except there should be either two numbers per row or one, 
    which will become a range from and to that value.
    """
    entries = read_id_integers(handleish, sep, header)
    for i, entry in enumerate(entries):
        # a single number will be made a range only containing that position
        if len(entry[1]) == 1: 
            entries[i] = entry[0], [entry[1][0], entry[1][0]]
        else: assert len(entry[1]) == 2, "There should be exactly two numbers to a range."
    return entries



def read_ranges_force(handleish, sep=None, header=False):
    """
    Same as read_ranges, except we log errors instead of raising exceptions.
    """
    entries = read_id_integers(handleish, sep, header)
    for i, entry in enumerate(entries):
        # a single number will be made a range only containing that position
        if len(entry[1]) == 1: 
            entries[i] = entry[0], [entry[1][0], entry[1][0]]
        elif len(entry[1]) != 2:
            log.error("Not exactly 2 integers found ({}) for id:\t{}".format(entry[1], entry[0]))
            entries[i] = entry[0], None
    return entries


def _read_text_integers(handleish, sep=None, header=False):
    """
    Read and separate text and integers, where integers can be found in the form of ranges, e.g. "4-10"
    :param handleish: 
    :return: [[text,...],...], [[number,...],...]
    """
    texts, numbers = [], []
    
    with as_handle(handleish) as fh:
        if header: next(fh)
        
        for line in fh:
            line = line.strip().split(sep)
            texts.append([])
            numbers.append([])
            for cell in line:
                try: numbers[-1].append(int(cell))
                except ValueError: pass
                else: continue
                # match for range, i.e. numbers separated by one or two non-alphabet character.
                _range = st.re_range(cell)
                if _range is not None: numbers[-1].extend(_range)
                else: texts[-1].append(cell)
    
    return texts, numbers




