#!/usr/bin/env python3
import numpy as np
from scipy.stats import entropy
import pandas as pd
from Bio.File import as_handle
from degnutil.string_util import parse_floatint
from degnutil.input_output import log

"""
This script is for motif related functions. 
It is not meant to be run on its own, but rather support other scripts.
"""

def read_motifs(handleish, delimiter=None, skip='#'):
    """
    Read matrix/matrices from file.
    If multiple matrices are provided, also provide name for each motif to match against sequence IDs.
    File format: alignment position along rows, alphabet along columns.
    Scanning for numbers to make each matrix.
    Non-empty row with text is considered key for the following matrix. Last row is used if there are multiple text rows before a matrix, 
    except if the last row is alphabet, then key is second-to-last.
    If first column of matrix is int and the rest are float the first is assumed to be position index and is removing.
    empty lines and text lines separate each matrix.
    :param handleish: Motif file with PFM(s) (Position Frequency Matrix) or other matrix representation of sequence motifs.
    :param delimiter: delimiter character. Default=None meaning whitespace
    :param skip: skip lines starting with this character
    :return: numpy or pandas array, list of arrays, or dict with arrays
    """
    arrays = []
    text_lines, number_lines = [], []
    
    with as_handle(handleish) as infile:
        for line in infile:
            if line.startswith(skip): continue
            line = line.strip()
            # are we at an empty line separating entries?
            if not line:
                if number_lines:
                    arrays.append(_read_motif(text_lines, number_lines, delimiter))
                    text_lines, number_lines = [], []
                continue
            # are we at a row of numbers?
            try: [float(l) for l in line.split(delimiter)]
            except ValueError:
                if number_lines: # we just started reading a new text header
                    arrays.append(_read_motif(text_lines, number_lines, delimiter))
                    text_lines, number_lines = [], []
                text_lines.append(line)
            else:
                number_lines.append(line)
        # last entry in file
        arrays.append(_read_motif(text_lines, number_lines, delimiter))
    
    if len(arrays) == 1: return arrays[0][1]
    if all(named_array[0] is None for named_array in arrays): return arrays
    if any(named_array[0] is None for named_array in arrays):
        raise IOError("Some arrays in the motif file lacks naming.")
    return dict(arrays)


def _read_alphabet(text_lines, delimiter=None):
    """
    Scan for alphabet in text lines associated with a motif.
    :param text_lines: list of stripped strings.
    :param delimiter: delimiter character that splits cells in the file.
    :return: None if no valid alphabet found or list of single characters otherwise
    """
    if len(text_lines) == 0: return
    line = text_lines[-1]
    l = line.split(delimiter)
    if len(l) == 1:  # no delimiter characters found, might be given as a single string
        # in that case is must be uppercase to be valid
        if line.isalpha() and line.isupper(): return list(line)
        return
    # if given with delimiters it is allowed to be all lowercase
    # allow first cell to be position column name
    alphabet = l[1:]
    alphabet_str = "".join(alphabet)
    # alphabet has to be single characters
    if len(alphabet) == len(alphabet_str) and alphabet_str.isalpha() and (alphabet_str.isupper() or alphabet_str.islower()):
        # first cell is included as well if it is single character and has same case
        if len(l[0]) == 1 and l[0].isupper() == alphabet_str.isupper():
            return [l[0]] + alphabet
        return alphabet


def _read_array(number_lines, alphabet=None, delimiter=None):
    """
    Read a single motif entry
    :param number_lines: list of strings. First column might be position indices.
    :param alphabet: list of characters or None to potentially be used as column names
    :return: string name or None, numpy or pandas array
    """
    array = [ls.split(delimiter) for ls in number_lines]
    # is the first column integers?
    try: [int(row[0]) for row in array]
    # everything is floats
    except ValueError: array = np.asarray(array, dtype=float)
    else:
        # is the whole array actually integers or only the first column?
        try: [[int(cell) for cell in row] for row in array]
        # only the first column is integers and the rest are floats. Remove the first column.
        except ValueError: array = np.asarray([row[1:] for row in array], dtype=float)
        else:
            # the whole matrix is integers
            array = np.asarray(array, dtype=int)
            if alphabet is not None:
                # is it a perfect match or is the first column position indices?
                if len(alphabet) == array.shape[1]: return pd.DataFrame(array, columns=alphabet)
                if len(alphabet) == array.shape[1] - 1: return pd.DataFrame(array[:,1:], columns=alphabet)
            return array
    
    if alphabet is None or len(alphabet) != array.shape[1]: return array
    return pd.DataFrame(array, columns=alphabet)


def _read_motif(text_lines, number_lines, delimiter=None):
    """
    Read a single motif entry
    :param text_lines: list of strings. The text lines where the last line is name or second-to-last if the last line is alphabet
    :param number_lines: list of strings. First column might be position indices.
    :return: string name or None, numpy or pandas array
    """
    array = _read_array(number_lines, _read_alphabet(text_lines), delimiter)
    if len(text_lines) == 0: return None, array
    if isinstance(array, pd.DataFrame):  # means the last line was valid as alphabet
        if len(text_lines) == 1: return None, array
        return text_lines[-2], array
    return text_lines[-1], array


def estimate_n_samples(empPPM):
    """
    Empirical PPM = PFM / n_samples => PFM = PPM * n_samples.
    This function is used when n_samples is unknown. It is estimated from the smallest non-zero value. 
    :param empPPM: empirical Position Probability Matrix
    :return: float. estimated number of samples
    """
    # in order to handle both numpy and pandas array
    empPPM = np.asarray(empPPM)
    if empPPM.dtype != float:
        log.error("empirical PPM values should contain fractions, int indicates counts")
    # n_samples >= n_samples_est
    n_samples_est = 1 / min(empPPM[empPPM > 0])
    return n_samples_est


def empPPM2PFM(empPPM):
    """
    Empirical PPM = PFM / n_samples => PFM = PPM * n_samples.
    This function is used when n_samples is unknown. It is estimated from the smallest non-zero value. 
    :param empPPM: empirical Position Probability Matrix
    :return: PFM (Position Frequency Matrix)
    """
    return empPPM * estimate_n_samples(empPPM)


def PFM2PPM(PFM, bg_freq=None, alpha=0.0001):
    """
    Get PPM from PFM using pseudo counts, which is a method of estimating true counts that would be observing with a larger sample size.
    It avoid counts that are zero and log likelihoods that are -inf, both of which are problematic.
    :param PFM: numpy int 2D array. Counts of each possible observation along axis=1, performed in parallel along axis=0.
    :param bg_freq: list, dict or numpy array. background frequencies if they are to be considered.
    :param alpha: float
    :return: numpy array. PPM
    """
    n_samples = sum(np.asarray(PFM)) / PFM.shape[0]
    if bg_freq is None:
        # simple method from wiki which doesn't consider background frequencies 
        # https://en.wikipedia.org/wiki/Additive_smoothing#Pseudocount
        return (PFM + alpha) / (n_samples + alpha * PFM.shape[1])

    # method that considers background frequencies from http://dx.doi.org/10.6064/2012/917540
    try: # numpy or pandas
        pseudo_freq = alpha * bg_freq
    except TypeError:
        if isinstance(bg_freq, dict):
            assert isinstance(PFM, pd.DataFrame), "If background frequencies are dict, then PFM should have column names"
            pseudo_freq = {k:alpha*v for k,v in bg_freq.items()}
            f_pseudo = sum(pseudo_freq.values())
        else: # list-like collection
            print(bg_freq)
            pseudo_freq = alpha * np.asarray(bg_freq)
            f_pseudo = sum(pseudo_freq)
    else:
        try: f_pseudo = sum(pseudo_freq)
        except TypeError: # was pandas
            assert len(pseudo_freq) == 1, "background frequencies given in pandas table should only contain 1 row"
            f_pseudo = pseudo_freq.sum(axis=1)
            pseudo_freq = dict(pseudo_freq.iloc[0])
    
    return (PFM + pseudo_freq) / (n_samples + f_pseudo)


def PPM2PWM(PPM, bg_freq=None):
    """
    Calculate Position Weight Matrix from Position Probability Matrix.
    http://dx.doi.org/10.6064/2012/917540
    :param PPM: numpy or pandas array or list. Position Probability Matrix.
    :param bg_freq: numpy 1d array or dict. background frequencies. Default is equal frequencies.
    :return: Position Weight Matrix, which is the log2(prob of letter at location / background prob of letter).
    """
    if bg_freq is None: bg_freq = np.repeat(1/PPM.shape[1], PPM.shape[1])
    elif type(bg_freq) is list: bg_freq = np.asarray(bg_freq)
    # same divisions performed on each row, meaning motif shape = (#positions, len(alphabet)) 
    # works for dict bg_freq and pandas PPM or numpy 1d array bg_freq with either pandas or numpy PPM
    return np.log2(PPM / bg_freq)


def read_meme(handleish):
    """
    MEME minimal format described here:
    http://meme-suite.org/doc/meme-format.html
    Reading MEME old format is implemented in Bio.motifs.meme but Bio.motifs.minimal is still under development, 
    so we write a basic replacement here.
    :param handleish: file handle or filename
    :return: {"alphabet":[letter,...], "strands":[character,...], "bg_freq":{letter:bg_freq,...}, "empPPM":{name:empPPM,...}, "n_samples":{name:n_samples,...}} 
    """
    alphabet, strands, bg_freq, matrices, n_samples = None, None, None, {}, {}
    name = None
    
    with as_handle(handleish) as infile:
        if not next(infile).startswith("MEME version"):
            raise ValueError("File not in MEME format")
        
        for line in infile:
            line = line.strip()
            if not line: continue
            
            if line.startswith("ALPHABET="):
                alphabet = line[len("ALPHABET="):].lstrip()
                # minimal meme format allows for multi-line alphabets. Let's not worry about that if we don't have to.
                if not alphabet.isalpha(): 
                    raise NotImplementedError("Multi-line alphabet descriptions not supported yet.")
                alphabet = list(alphabet)
            
            elif line.startswith("strands:"):
                strands = line.split()[1:]
            
            elif line.startswith("Background letter frequencies"):
                bg_freq = {}
                for bg_freq_line in infile:
                    bg_freq_line = bg_freq_line.strip()
                    if not bg_freq_line: break
                    if bg_freq_line.startswith("MOTIF"):  # read past bg freq
                        name = bg_freq_line.split()[1]  # ignore alternative names
                        break

                    bg_freq_line = bg_freq_line.split()
                    # append to dict, handles multi-line description
                    bg_freq = {**bg_freq, **dict(zip(bg_freq_line[::2], bg_freq_line[1::2]))}
            
            elif line.startswith("MOTIF"):
                name = line.split()[1]  # ignore alternative names
            
            elif line.startswith("letter-probability matrix:"):
                mat_info = line[len("letter-probability matrix:"):].lstrip().split()
                mat_info = dict([(k.rstrip('='), parse_floatint(v)) for k,v in zip(mat_info[::2], mat_info[1::2])])
                if "alength" in mat_info and alphabet is not None:
                    assert mat_info["alength"] == len(alphabet)
                if "nsites" in mat_info:
                    n_samples[name] = mat_info["nsites"]
                # read the matrix numbers
                matrices[name] = []
                for mat_line in infile:
                    mat_line = mat_line.strip()
                    if not mat_line: break
                    if mat_line.startswith("MOTIF"):
                        name = mat_line.split()[1]  # ignore alternative names
                        break
                    matrices[name].append(mat_line.split())
                    if "alength" in mat_info:
                        assert len(mat_line.split()) == mat_info["alength"]
    
    # parse text in mats
    for k,v in matrices.items():
        matrices[k] = np.asarray(v, dtype=float)
    # assign column names
    if alphabet is not None:
        for k,v in matrices.items():
            matrices[k] = pd.DataFrame(v, columns=alphabet)
    
    return {"alphabet": alphabet, "strands": strands, "bg_freq": bg_freq, "matrices": matrices, "n_samples": n_samples}


def write_meme(handleish, matrices, alphabet=None, strands=None, bg_freq=None, n_samples=None):
    """
    Write minimal MEME version 4 format.
    :param handleish: 
    :param matrices: dict of arrays (empirical probability matrices)
    :param alphabet: str or list of characters. Can be found from bg_freq letters
    If list, then alphabet should be provided.
    :param strands: str or list of strands, e.g. ["+"], "+-", ...
    :param bg_freq: list or dict of background frequencies for each letter in the alphabet. 
    :param n_samples: list, dict or int. number of observations used in creation of each motif.
    :return: 
    """
    if alphabet is None:
        if bg_freq is not None: alphabet = bg_freq.keys()
        else:
            # try to alphabet from motifs
            for array in matrices.values():
                if isinstance(array, pd.DataFrame):
                    alphabet = array.columns
                    break
    
    if alphabet is not None:
        # final format
        alphabet = ''.join(alphabet)
        # motifs should have the same alphabet
        for array in matrices.values():
            if isinstance(array, pd.DataFrame):
                assert alphabet == ''.join(array.columns), "Motifs have different alphabets"
    
    if isinstance(bg_freq, list):
        assert alphabet is not None, "Alphabet should be provided in order to write the background frequencies"
        bg_freq = {k:v for k,v in zip(alphabet, bg_freq)}
    
    # convert n_samples to dict one way or the other
    if n_samples is None: n_samples = {}
    elif isinstance(n_samples, list): n_samples = {name: n for name, n in zip(matrices, n_samples)}
    elif isinstance(n_samples, int): n_samples = {name: n_samples for name in matrices}
    else: assert isinstance(n_samples, dict), "unrecognized n_samples type"
    
    with as_handle(handleish, 'w') as fh:
        fh.write("MEME version 4\n\n")
        if alphabet is not None: fh.write("ALPHABET= {}\n\n".format(alphabet))
        if strands is not None: fh.write("strands: {}\n\n".format(" ".join(strands)))
        if bg_freq is not None:
            fh.write("Background letter frequencies\n")
            fh.write(" ".join(" ".join(map(str,kv)) for kv in bg_freq.items()) + "\n\n")
        
        for name, array in matrices.items():
            w, alength = array.shape
            fh.write("MOTIF {}\n".format(name))
            fh.write("letter-probability matrix: alength= {} w= {}".format(alength, w))
            try: nsites = int(n_samples[name])
            except KeyError: fh.write("\n")
            else: fh.write(" nsites= {}\n".format(nsites))
            for row in np.asarray(array): fh.write(" " + " ".join(map(str,row)) + "\n")
            fh.write("\n")


def seqs2PFM(seqs, alphabet):
    """
    Get frequency of each letter at each position (PFM) from an array of sequences.
    :param seqs: 2D array with size (sequences, sequence_length)
    :param alphabet: iterable of letters
    :return: 2D array with shape (sequence_length, alphabet)
    """
    return np.asarray([np.sum(seqs == l, axis=0) for l in alphabet]).T


def seqs2PFM_weighted(seqs, alphabet, weight):
    """
    Get frequency of each letter at each position (PFM) from an array of sequences.
    :param seqs: 2D array with size (sequences, sequence_length)
    :param alphabet: iterable of letters
    :param weight: vector of shape (sequences) with weight for each sequence
    :return: 2D array with shape (sequence_length, alphabet)
    """
    weight = weight.reshape(-1, 1)
    return np.asarray([np.sum((seqs == l) * weight, axis=0) for l in alphabet]).T


def get_IC(f):
    alphabet_size = f.shape[1]
    return np.asarray([np.log2(alphabet_size) - entropy(f[i, :], base=2) for i in range(alphabet_size)])

def small_sample_correction(n_samples, alphabet_size):
    return 1 / np.log(2) * (alphabet_size - 1) / (2*n_samples)

def logo_height(f, n_samples):
    ic = get_IC(f) - small_sample_correction(n_samples, f.shape[1])
    return ic * f





