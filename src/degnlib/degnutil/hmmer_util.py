import numpy as np
from Bio import File
from degnutil.string_util import parse_value

# since hmmer seems not be read properly with python Bio package we have to write some code for that

def parse_hmmer(handleish):
    hmm = {}
    with File.as_handle(handleish) as handle:
        # read first line, e.g. "HMMER3/f [3.1b1 | February 2013]"
        hmm["version"] = next(handle).strip().split()[0]
        # reading header fields
        for line in handle:
            l = line.strip().split()
            if l[0] == "HMM":
                alphabet = l[1:]
                break
            hmm = {**hmm, **_parse_hmmer_header(l)}
        
        # reading main body       
        # skip line with "m->m     m->i     m->d     i->m     i->i     d->m     d->d"
        next(handle)
        hmm["compo"] = [float(v) for v in next(handle).strip().split()[1:]]
        # next two lines not supported yet
        next(handle); next(handle)
        hmm["match_emission"]  = []
        hmm["insert_emission"] = []
        for i, line in enumerate(handle):
            line = line.strip()
            if line == "//": return hmm
            l = line.split()
            # make sure the model position number makes sense
            assert int(l[0]) == i+1
            hmm["match_emission"].append(_parse_scores(l[1:1+len(alphabet)]))
            insert_emission_line = next(handle).strip().split()
            hmm["insert_emission"].append(_parse_scores(insert_emission_line[:len(alphabet)]))
            # state transition lines not supported yet
            next(handle)

def _parse_hmmer_header(fields):
    name = []
    for i in range(len(fields)):
        if fields[i].isupper(): name.append(fields[i])
        else:
            value = [parse_value(v) for v in fields[i:]]
            if len(value) == 1: value, = value
            if value == "no": value = False
            if value == "yes": value = True
            return {" ".join(name): value}

def _parse_scores(scores):
    """
    probabilities are given as score = -ln(p)
    :param scores: list of string scores
    :return: numpy float probabilities 
    """
    return np.exp(-np.asarray(scores, dtype=float))

