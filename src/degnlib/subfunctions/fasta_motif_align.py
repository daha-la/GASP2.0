#!/usr/bin/env python3
import argparse
from degnutil import argument_parsing as argue, motif_util as mot, read_write as rw
from Bio import SeqIO, File
from Bio.Seq import Seq
from Bio.Data.IUPACData import protein_letters
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

def get_parser():
    parser = argparse.ArgumentParser(
        description="Align sequences to motifs. "
                    "Ungapped alignment, i.e. the sequences will not be broken into pieces. "
                    "Each sequence is simply moved along each position of the motif for best match. ")
    argue.inoutfile(parser)
    argue.delimiter(parser)
    parser.add_argument("-m", "--motif", required=True, 
                        help="Motif file with PFM(s) (Position Frequency Matrix), e.g. simply a whitespace delimited matrix of numbers. "
                             "Optionally provide name for each motif to match against sequence IDs. "
                             "File format: alignment position along rows, alphabet along columns. "
                             "Scanning for numbers to make the PFM(s), closest text row before each PFM belongs to the PFM (unless it is alphabet). "
                             "MEME format can also be used.")
    parser.add_argument("-b", "--bg-freq", nargs="+", default=None,
                        help="Background frequencies for each letter of the alphabet. "
                             "Can be values given directly or as a file."
                             "Default is equal frequency.")
    parser.add_argument("-a", "--alphabet",
                        help="Alphabet to use. "
                             "Can also be provided for each PFM as column names (first priority)."
                             "Default is sorted standard DNA, RNA, or protein alphabets, depending on PFM size.")
    parser.add_argument("-t", "--thres", dest="threshold", type=float, 
                        help="PWMS (log likelihood ratio) threshold. Default is no threshold.")
    return parser


def parse_args(args):
    args = argue.parse_inoutfile(argue.parse_delimiter(args))
    
    if args.bg_freq is not None:
        if len(args.bg_freq) == 1:
            args.bg_freq = rw.read_vector(args.bg_freq[0], args.delimiter)
        else:
            args.bg_freq = map(float, args.bg_freq)
    
    return args


def get_PWM(motif, bg_freq=None):
    """
    Get a PWM from a PFM or empirical PPM.
    motif is assumed to be PFM if it contains int, and PPM if it contains floats.
    :param motif: PFM or empirical PPM
    :param bg_freq: background frequencies
    :return: PWM, which is log2(prob of letter at location / background prob of letter)
    """
    try: is_empPPM = motif.dtype == float
    except AttributeError:
        is_empPPM = all(motif.dtypes == float)
        assert all((motif.dtypes == float) == is_empPPM), "inconsistent dtypes in motif pandas array"
    if is_empPPM:
        motif = mot.empPPM2PFM(motif)
    motif = mot.PFM2PPM(motif, bg_freq, alpha=0.1)
    motif = mot.PPM2PWM(motif, bg_freq)
    return motif


def align(record, PWM, alphabet=None):
    """
    Ungapped alignment of sequence to motif.
    :param record: SeqRecord with unaligned sequence
    :param PWM: numpy or pandas array
    :param alphabet:
    :return: SeqRecord with aligned sequence, float PWMS
    """
    if isinstance(PWM, pd.DataFrame):
        alphabet = PWM.columns
        PWM = np.asarray(PWM)

    seq_len = len(record.seq)
    motif_len = len(PWM)
    
    if isinstance(alphabet, str):
        alphabet = list(alphabet)
    elif alphabet is None:
        # unique and sorted
        alphabet = np.unique(list(record.seq))
        if len(alphabet) < PWM.shape[1]:
            # assume standard DNA, RNA or protein
            if PWM.shape[1] == 20: alphabet = protein_letters
            elif PWM.shape[1] == 4:
                if "U" in record.seq: alphabet = "ACGU"
                else: alphabet = "ACGT"
        elif len(alphabet) > PWM.shape[1]:
            raise ValueError("More unique characters in sequence that motif size.")
    
    sequence = np.asarray(list(record.seq))
    js = np.zeros(seq_len, dtype=int)
    for j, letter in enumerate(alphabet):
        js[sequence == letter] = j
    
    vs = np.ones(seq_len, dtype=bool)
    
    if PWM.shape[0] < seq_len:
        raise NotImplementedError("Sequence to align shorter than motif.")
    
    best_PWMS, best_offset = -np.inf, 0
    for offset in range(motif_len - seq_len + 1):
        _is = np.arange(seq_len)+offset
        # log of likelihood ratio or log-odds http://dx.doi.org/10.6064/2012/917540
        PWMS = sum(PWM[coo_matrix((vs, (_is, js)), shape=PWM.shape).toarray()])
        if PWMS > best_PWMS:
            best_PWMS = PWMS
            best_offset = offset
    
    record.seq = np.repeat("-", motif_len)
    record.seq[best_offset:best_offset+seq_len] = sequence
    record.seq = Seq("".join(record.seq))
    
    return record, best_PWMS


def main(args):
    args = parse_args(args)
    try: meme = mot.read_meme(args.motif)
    except ValueError: motif = mot.read_motifs(args.motif, args.delimiter)
    else:
        motif = meme["matrices"]
        # potentially convert to PFM, this will mean it will not have sample size estimated in get_PWM later
        if meme["n_samples"]: motif *= meme["n_samples"]
        if args.alphabet is None: args.alphabet = meme["alphabet"]
        if args.bg_freq is None: args.bg_freq = meme["bg_freq"]
        
    
    # get PWM(s) from the provided PFM(s) or empirical PPM(s)
    if isinstance(motif, dict):
        for motif_name, motif_array in motif.items():
            motif[motif_name] = get_PWM(motif_array, args.bg_freq)
    elif isinstance(motif, np.ndarray) or isinstance(motif, pd.DataFrame):
        motif = get_PWM(motif, args.bg_freq)
    else:
        raise NotImplementedError("Motif wasn't read as either dict or array.")
    
    if args.threshold is None: args.threshold = -np.inf
    
    with File.as_handle(args.outfile, 'w') as outfile:
        if isinstance(motif, dict):
            for record in SeqIO.parse(args.infile, "fasta"):
                record, PWMS = align(record, motif[record.id], args.alphabet)
                if PWMS > args.threshold: SeqIO.write(record, outfile, "fasta")
        elif isinstance(motif, np.ndarray) or isinstance(motif, pd.DataFrame):
            for record in SeqIO.parse(args.infile, "fasta"):
                record, PWMS = align(record, motif, args.alphabet)
                if PWMS > args.threshold: SeqIO.write(record, outfile, "fasta")


if __name__ == '__main__':
    main(get_parser().parse_args())
