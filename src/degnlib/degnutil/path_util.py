#!/usr/bin/env python3
import os
import re
import numpy as np
from pathlib import Path

"""
This script is for path functions, it should act as an extension of the pathlib and os library.
"""


def exists(path):
    """
    Long names can cause OSError, and None can cause TypeError, 
    so if "path" might not even be remotely like a path this can be useful to ignore.
    :param path: 
    :return: 
    """
    try: return Path(path).exists()
    except (OSError, TypeError): return False


def isfile(path):
    """
    Long names can cause OSError, and None can cause TypeError, 
    so if "path" might not even be remotely like a path this can be useful to ignore.
    :param path: 
    :return: 
    """
    try: return Path(path).is_file()
    except (OSError, TypeError): return False
    

def any_exists(paths):
    """
    return whether any of the given paths exists
    :param paths: 
    :return: 
    """
    return any(Path(p).exists() for p in paths)


def has_suffix(path, suffixes):
    """
    Find out if a path has one of the given array_suffixes (extensions)
    :param path: Path or string
    :param suffixes: one or more string array_suffixes
    :return: bool
    """
    # no need to check if there is only a single suffix in "array_suffixes"
    return Path(path).suffix in suffixes


def match(pattern, paths=None):
    """
    Get path(s) of file(s) that match(es) the given pattern.
    :param pattern: string pattern
    :param paths: paths to match against the prefix, ignore to use files in cwd
    :return: list of string path(s)
    """
    # using cwd?
    if paths is None: return list(Path.cwd().glob(pattern))
    else: return [p for p in paths if Path(p).match(pattern)]


def match_prefix(prefix, paths=None):
    """
    Get path(s) of file(s) that match(es) the given prefix.
    :param prefix: string start of filename
    :param paths: paths to match against the prefix, ignore to use files in cwd
    :return: list of string path(s)
    """
    return match(prefix + '*', paths)


def match_suffix(suffix, paths=None):
    """
    Get path(s) of file(s) that match(es) the given suffix (file extension).
    :param suffix: string full suffix(es) (file extension(s))
    :param paths: paths to match against the suffix, ignore to use files in cwd
    :return: list of string path(s)
    """
    try: return match('*' + suffix, paths)
    # multiple suffixes given
    except TypeError:
        if paths is None: paths = os.listdir()
        return [p for p in paths if has_suffix(p, suffix)]


def shortest_match_prefix(prefix, paths=None):
    """
    Get path(s) of file(s) that match(es) the given prefix.
    Two matches are equal if they have the same length excluding the suffix (file extension),
    otherwise the shorter is returned.
    :param prefix: string start of filename
    :param paths: paths to match against the prefix, ignore to use files in cwd
    :return: list of string path(s)
    """
    paths = match_prefix(prefix, paths)
    if len(paths) == 0: raise FileNotFoundError("No files matched the given prefix")
    if len(paths) == 1: return paths
    # find best matches (shortest matches)
    stem_lens = [len(p.stem) for p in paths]
    return np.asarray(paths)[np.asarray(stem_lens) == min(stem_lens)]


def increment(path, digits=2):
    """
    Get path(s) with a number appended for given path(s). The number increments to avoid file conflict.
    :param path: path or paths if to make sure all are given the same number.
    :param digits: number of digits for the number. If the numbering exeeds this, an error is raised.
    :return: path(s)
    """
    is_single = not isinstance(path, list)
    if is_single: path = [path]
    paths = [Path(p) for p in path]
    
    for i in range(10**digits):
        new_paths = [p.with_name("{}_{}{}".format(p.stem, str(i).rjust(digits, '0'), p.suffix)) for p in paths]
        if any(p.exists() for p in new_paths): continue
        
        if is_single: return new_paths[0]
        return new_paths
    
    raise FileExistsError("Incrementing filename to avoid conflict failed since there are too many files")


def get_increment(path):
    """
    Return the increment value in a filename.
    Assuming file will be named as <path.stem>_<increment number><path.suffix>
    :param path: string or Path
    :return: int increment number, -1 if no increment value found
    """
    try: return int(Path(path).stem.split("_")[-1])
    except ValueError: return -1


def glob(pattern):
    """
    A glob that uses re regex instead of the more limited glob regex.
    :param pattern: str pattern
    :return: list of files.
    """
    return list(iglob(pattern))


def iglob(pattern):
    """
    A glob that uses re regex instead of the more limited glob regex.
    :param pattern: str pattern
    :return: iterator of files.
    """
    for f in Path(pattern).parent.iterdir():
        if re.match(pattern, str(f)): yield f


def get_increments(unincremented):
    """
    Get all the increment versions of an unincremented path.
    :param unincremented: str without increment
    :return: list of str
    """
    path = Path(unincremented)
    pattern = str(path.parent / Path(path.stem)) + "_([0-9])+" + path.suffix
    return glob(pattern)


def get_highest_increment(unincremented):
    """
    Return the newest increment for a filename.
    Assuming increments will be named as <path.stem>_<increment number><path.suffix>
    :param unincremented: string or Path that is the unincremented version
    :return: incremented filename of file that exists
    """
    out = unincremented
    highest_inc = -1
    for filename in get_increments(unincremented):
        inc = get_increment(filename)
        if inc > highest_inc:
            highest_inc = inc
            out = filename
    return out
    

