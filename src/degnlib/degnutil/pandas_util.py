#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
import re

"""
This script is for utilities related to pandas tables.
Some functions are typical functions that are made to work the same on both numpy and pandas.
"""


def read_pandas(handleish, sep=None, header='infer'):
    """
    Simple wrapper for convenience
    :param handleish: 
    :param sep: default whitespace
    :param header: default infer header names but assume there is a header
    :return: pandas DataFrame
    """
    if sep is None: sep = '\s+'
    if header is True: header = 0
    elif header is False: header = None
    return pd.read_table(handleish, sep=sep, header=header)


def write_pandas(handleish, df, sep="\t", header=None):
    if sep is None: sep = "\t"
    if header is None: header = has_columns(df)
    df.to_csv(handleish, sep=sep, header=header, index=has_rownames(df))


def read_id_integers(handleish, sep=None, header='infer'):
    """
    Similar to function with same name in read_write except here we use pandas in order to return a pandas DataFrame.
    It makes it easier to deal with headers in the case of being flexible to handle either stream of file.
    :param handleish: 
    :param sep: 
    :param header: 
    :return: [id,...], [[int,...],...], id column name
    """
    df = read_pandas(handleish, sep, header)
    text_column = None
    number_columns = []
    
    for column, dtype in zip(df, df.dtypes):
        if is_integer_dtype(dtype):
            number_columns.append(df[column])
            continue
        
        # match for intervals, i.e. a number in the beginning and end of cell.
        starts, ends = [], []
        for cell in df[column]:
            start, end = re.match("\d+", cell), re.findall("\d+$", cell)
            if start is not None and len(end) == 1:
                starts.append(int(start[0]))
                ends.append(int(end[0]))
            else: break
        
        # convert to intervals
        if len(starts) == len(df):
            number_columns.append(starts)
            number_columns.append(ends)
        elif text_column is None:  # only the first text column found
            text_column = column
    
    assert text_column is not None, "No text column found"
    # if header is either None or False it should return None
    return df[text_column], list(zip(*number_columns)), None if not header else text_column



def sum(array):
    """
    Sum over all values in either a numpy or pandas array
    :param array: numpy or pandas array
    :return: numpy number sum
    """
    try: return array.sum(skipna=True).sum()
    except TypeError: return array.sum()


def min(array, axis=None):
    try: return array.values.min(axis)
    except AttributeError: return array.min(axis)


def max(array, axis=None):
    try: return array.values.max(axis)
    except AttributeError: return array.max(axis)


def mean(array, axis=None):
    try: return array.values.mean(axis)
    except AttributeError: return array.mean(axis)


def median(array, axis=None):
    try: return np.median(array.values, axis)
    except AttributeError: return np.median(array, axis)


def std(array, axis=None):
    try: return array.values.std(axis)
    except AttributeError: return array.std(axis)


def index(array, idx):
    """
    Returns what you would like array[index] to return, but it doesn't when mixing pandas and numpy arrays.
    :param array: numpy or pandas array
    :param idx: numpy or pandas array
    :return: array with same format as input array
    """
    if is_pandas(array) and not is_pandas(idx):
        return array[pd.DataFrame(idx, columns=array.columns, index=array.index)]
    if not is_pandas(array) and is_pandas(idx):
        return array[idx.values]
    return array[idx]


def concat(arrays):
    """
    Concat arrays assumed to be 2D along axis 0.
    :param arrays: collection of pandas DataFrames or numpy arrays
    :return: pandas DataFrame or numpy array
    """
    try: return pd.concat(arrays)
    except TypeError: return np.concatenate([np.atleast_2d(a) for a in arrays])


def concat_pandas(arrays, group=""):
    """
    Concat pandas tables by adding extra rows.
    Optionally a new column is added differentiating between each table.
    :param arrays: collection of pandas arrays, if dict, the keys can be used to differentiate them as new column.
    :param group: string name of the new column that groups tables. Ignore to not add.
    :return: 
    """
    if not group: return pd.concat(arrays)
    return pd.concat(arrays, names=[group]).reset_index(level=0)


def col(table, column):
    """
    Get the column of table either specified by key or int index
    :param table: pandas dataframe
    :param column: column name or column int index
    :return: column
    """
    try: return table[column]
    except KeyError: return table.iloc[:, int(column)]


def colID2name(df, ids):
    """
    Get column name(s) given either column names or integer column index
    :param df:
    :return:
    """
    if np.isscalar(ids): return colID2name(df, [ids])[0]
    ids = list(ids)
    for i, _id in enumerate(ids):
        if _id not in df:
            try: ids[i] = df.columns[int(_id)]
            except (ValueError, IndexError):
                raise IndexError(f"{_id} not a column in dataframe.")
    return ids


def guess_col(table, column_options):
    """
    Get a column from a table flexibly.
    :param table: numpy array or pandas
    :param column_options: list of possible column names or int indexes to try.
    :return: column
    """
    if is_pandas(table):
        for column_option in column_options:
            if column_option in table:
                return col(table, column_option)
    else:
        for column_option in column_options:
            try: return table[:, column_option]
            except IndexError: pass
            

def get_columns(array):
    """
    Get column names from an array assumed to be a 2D table
    :param array: pandas or numpy
    :return: Index or list
    """
    try: return array.columns
    except AttributeError: return list(range(np.atleast_2d(array).shape[1]))


def has_columns(df):
    # might not always work
    try: return not isinstance(df.columns, pd.RangeIndex)
    except AttributeError: return False


def has_rownames(df):
    try: return not isinstance(df.index, pd.RangeIndex)
    except AttributeError: return False


def is_fully_named_pandas(array):
    """
    Return whether array is a pandas DataFrame with both index and column names 
    which are not the default RangeIndex.
    :param array: 
    :return: bool
    """
    try:
        return not isinstance(array.index, pd.RangeIndex) and \
               not isinstance(array.columns, pd.RangeIndex)
    except AttributeError:
        return False


def to_pandas(array, index=None, columns=None, digits=3):
    """
    Create a pandas array optionally using incrementing index and column names
    :param array: numpy array or pandas array
    :param index: string index identifier or list of index names
    :param columns: string column identifier or list of column names
    :param digits: int number of digits to use for incremented naming
    :return: pandas DataFrame
    """
    l_index, l_columns = array.shape
    if index is not None:
        try:
            index = pd.Index(index)
        except TypeError:
            index = [index + "{:0{digits}d}".format(i, digits=digits) for i in range(l_index)]
    if columns is not None:
        try:
            columns = pd.Index(columns)
        except TypeError:
            columns = [columns + "{:0{digits}d}".format(i, digits=digits) for i in range(l_columns)]

    try:  # make sure to handle the case of array being pandas
        array.index = index
        array.columns = columns
    except AttributeError:
        array = pd.DataFrame(array, index=index, columns=columns)
    finally:
        return array


def is_pandas(array):
    return hasattr(array, "columns")


def remove_column_index(array, index):
    """
    Drop a range of columns from a pandas dataframe
    :param array: pandas dataframe
    :param index: iterable e.g. range or list or 1D array
    :return: array with the columns determined by index removed
    """
    columns = np.asarray(array.columns)[index]
    return array.drop(columns, axis=1)


def remove_row_index(array, index):
    """
    Drop a range of rows from a pandas dataframe
    :param array: pandas dataframe
    :param index: iterable e.g. range or list or 1D array
    :return: array with the rows determined by index removed
    """
    rows = np.asarray(array.index)[index]
    return array.drop(rows, axis=0)


def insert_column(array, index, column, value):
    """
    Insert a column in a pandas dataframe
    :param array: pandas dataframe
    :param index: position to insert the column in
    :param column: string name of column to insert
    :param value: value to insert in the row
    :return: pandas dataframe
    """
    # we want to not do inline to be consistent with the insert_row function
    array = array.copy()
    array.insert(index, column, value)
    return array


def insert_row(array, index, row, value):
    """
    Insert a row in a pandas dataframe
    :param array: pandas dataframe
    :param index: position to insert the row in
    :param row: string name of row to insert
    :param value: value to insert in the row
    :return: pandas dataframe
    """
    # there's something weird happening if you use "..." as pandas index
    if row == "...": row = ".."
    insert = pd.DataFrame(value, columns=array.columns, index=[row])
    return pd.concat((array.iloc[:index, ], insert, array.iloc[index:, ]))


def iloc(arr):
    """
    Like iloc of pandas but also works if we are using a numpy array
    :param arr: numpy or pandas
    :return: 
    """
    try: return arr.iloc
    except AttributeError: return arr
