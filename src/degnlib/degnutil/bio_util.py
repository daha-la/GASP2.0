#!/usr/bin/env python3

"""
This script is for utilities related to bio, specifically the Bio package.
"""
import sys
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.Data.SCOPData import protein_letters_3to1
from Bio.File import as_handle
import numpy as np
from pathlib import Path
from degnutil import string_util as st
import copy


def three2one(record:SeqRecord, undef_code='X'):
    """
    in-place
    :param record: 
    :param undef_code: 
    :return: 
    """
    seq = str(record.seq)
    seq = [seq[i:i+3] for i in range(0, len(seq), 3)]
    record.seq = Seq(''.join(_three2one(seq, undef_code)), record.seq.alphabet)
    return record


def _three2one(seq3, undef_code='X'):
    """
    
    :param seq3: list of 3 letter codes 
    :param undef_code: code to use for unknown 3 letter codes
    :return: list of 1 letter codes
    """
    seq1 = [protein_letters_3to1.get(c.upper(), undef_code) for c in seq3]
    if 'X' in seq1:
        unknowns = np.unique([seq3[i] for i in range(len(seq3)) if seq1[i] == 'X'])
        sys.stderr.write("Unknown codes: " + str(unknowns) + '\n')
    return seq1


def sequence(seq, id, name="", description="", dbxrefs=None, features=None, annotations=None, letter_annotations=None):
    """
    Better default values. Can handle a string sequence.
    :param seq: 
    :param id: 
    :param name: 
    :param description: 
    :param dbxrefs: 
    :param features: 
    :param annotations: 
    :param letter_annotations: 
    :return: SeqRecord object.
    """
    # make sure seq is converted from string to Seq object if it is not already done.
    try: seq = Seq(seq)
    except TypeError: pass
    return SeqRecord(Seq, id, name, description, dbxrefs, features, annotations, letter_annotations)


def slice_seq(record, start=None, stop=None):
    """
    Annotations are lost with normal slicing.
    :param record: 
    :param start: default=from start of sequence
    :param stop:  default=until end of sequence
    :return: sliced SeqRecord with annotations
    """
    record = copy.deepcopy(record)
    record.seq = record.seq[start:stop]
    return record


def subslice_1letter(string, start, end, origin=0, pad=0, assertion=True):
    """
    Like the subslice function, but allows for positions like "K4" where position 4 is used and it is asserted that the letter at that position is K.
    :param string:
    :param start:
    :param end:
    :param origin:
    :param pad:
    :param assertion: assert if the letter is correct
    :return:
    """
    # these try statements will raise a ValueError for int("") so if only a single letter is given as a position then that can be caught outside this call.
    try: start_letter, start = start[0], int(start[1:])
    except TypeError: start_letter = None
    try: end_letter, end = end[0], int(end[1:])
    except TypeError: end_letter = None
    slc = st.subslice(string, start, end, origin, pad)
    if assertion:
        if start_letter is not None:
            assert string[slc.start] == start_letter, f"{start_letter} not at (zero-indexed) pos {slc.start} in {string}"
        if end_letter is not None:
            assert string[slc.stop - 1] == end_letter, f"{end_letter} not at (zero-indexed) pos {slc.stop - 1} in {string}"
    return slc


def get_annotation(record, ID=None, default=None):
    """
    Get record.id, record.description, or a record annotation.
    :param record: 
    :param ID: 
    :param default:
    :return: 
    """
    try: return record.annotations[ID]
    except KeyError: pass
    try: return getattr(record, ID)
    except TypeError: return record.id  # ID is None
    except AttributeError:
        if default is None: raise KeyError("Annotation not found")
    return default


def get_record_id(record, ID=None):
    """
    Get id as a named annotation with fallback to "id"
    :param record:
    :param ID:
    :return:
    """
    try: return record.annotations[ID]
    except KeyError: return record.id


def ungap(seq, gap='-'):
    """
    Get an ungapped copy of a sequence by removing gap characters and converting to uppercase 
    :param seq: SeqRecord 
    :param gap: str. each character is removed from seq 
    :return: ungapped uppercase copy of seq
    """
    seq = seq.upper()
    for g in gap: seq.seq = seq.seq.ungap(g)
    return seq



def insertion_index(consensus):
    """
    Get index of insertions
    :param consensus: str, Seq or SeqRecord. Contains only 'x' and '.' for consensus and rest
    :return: index of '.' inside consensus
    """
    consensus = np.char.asarray(list(consensus))
    # make index for what is consensus
    idx = consensus == 'x'
    assert all(consensus[~idx] == '.'), "Consensus sequence not recognized"
    start, end = np.where(idx)[0][[0, -1]]
    # make reverse idx of insertions
    idx[0:start] = True
    idx[end + 1:len(idx)] = True
    return ~idx


### IO ###


def read(handle):
    """
    Read sequences and full sequence names as two lists
    :param handle: filename or stream
    :return: [seq,...], [description,...]
    """
    from Bio.SeqIO import parse
    return zip(*((r.seq, r.description) for r in parse(handle, "fasta")))


def write(seq, description, handle):
    """
    Write a fasta format entry.
    :param seq: 
    :param description: 
    :param handle: 
    :return: 
    """
    handle.write(">{}\n{}\n".format(description, seq))


def write_record(record, handle):
    write(str(record.seq), record.description, handle)


def read_pdb(filename):
    return PDBParser().get_structure(Path(filename).stem, filename)


def yield_pdb_sequences(structure):
    """
    Get the sequence, chain name and residue index for the first letter.
    hetflag id[0] == ' ' means it is not water ('W') or a heterogen residue (regex 'H.*')
    https://www.wwpdb.org/documentation/file-format-content/format33/sect4.html
    :param structure: pdb structure
    :return: iterator, each element is (Seq seq, string chain, int origin, string alphabet)
    """
    ppb = PPBuilder()
    for chain in structure.get_chains():
        pps = ppb.build_peptides(chain)
        if len(pps) == 0:  # not amino acids
            seq = [res.resname.strip().ljust(3) for res in chain]
            waters = np.asarray(seq)=='HOH'
            n_waters = sum(waters)
            # remove water from end, it should not be found other places
            assert n_waters == sum(waters[len(seq)-n_waters:len(seq)]), "Not all waters are at the end"
            seq = _three2one(seq[0:len(seq) - n_waters])
            # .rstrip(X) to remove trailing special atoms, e.g. MG, CA, ...
            yield ''.join(seq).rstrip('X'), chain.id, chain.child_list[0].id[1], 'nucl'
        else:
            # position of last residue in last peptide
            end = pps[-1][-1].id[1]
            seq = np.asarray(['-'] * end)
            for pp in pps:
                # -1 since it is 1-indexed
                pos = np.asarray([res.id[1] for res in pp]) - 1
                seq[pos] = list(pp.get_sequence())
            yield ''.join(seq).strip('-'), chain.id, pps[0][0].id[1], 'prot'


def read_selex(handleish):
    """
    Read a selex format file which is just a tab-separated file with name then sequence, 
    except sequences are potentially written in chunks, to avoid long lines
    :param handleish: filename or handle
    :return: ([str name, ...], [str seq, ...])
    """
    names, seqs = [], []
    with as_handle(handleish) as infile:
        # read first chunk of each record
        for line in infile:
            line = line.strip()
            if not line: break
            name, chunk = line.split()
            names.append(name)
            seqs.append(chunk)
        
        i = 0
        for line in infile:
            line = line.strip()
            if not line:
                i = 0
                continue
            name, chunk = line.split()
            assert name == names[i], "Name mismatch for selex format"
            seqs[i] += chunk
            i += 1
    
    return seqs, names
            

def yield_parsed_records(record, fieldsep=" ", kvsep="="):
    """
    Parse fields in seq description to their annotations dict.
    :param record: SeqRecord or [SeqRecord, ...]
    :param fieldsep: 
    :param kvsep: 
    :return: modified seq(s)
    """
    if isinstance(record, SeqRecord): record = [record]
    for seq in record:
        for field in seq.description.split(fieldsep):
            kv = field.split(kvsep)
            if len(kv) == 2:
                seq.annotations[kv[0]] = st.parse_value(kv[1])
        yield seq
        

def yield_deparsed_seqs(record, fieldsep=" ", kvsep="="):
    """
    Join annotation dicts into strings and put the ID first.
    :param record: 
    :param fieldsep: 
    :param kvsep: 
    :return: 
    """
    if isinstance(record, SeqRecord): record = [record]
    for seq in record:
        seq.description = seq.id
        annotation_string = fieldsep.join(f"{k}{kvsep}{v}" for k,v in seq.annotations.items())
        if annotation_string != "": seq.description += fieldsep + annotation_string
        yield seq
        

        
