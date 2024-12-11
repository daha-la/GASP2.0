#!/usr/bin/env python3
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import pandas as pd
import mdtraj as md
from Bio import AlignIO
import subprocess
from itertools import product

def get_parser():
    parser = argparse.ArgumentParser(description="""
    Encode enzyme using Structure-informed method. 
    Combines an MSA and a template to identify important residues and match them with every protein in MSA.
    Uses either sphere extraction or 
    """, formatter_class=RawTextHelpFormatter)

    parser.add_argument("-enc", "-encode", dest="encode",
        help="Determines the sequence encoding used for converting residues to numerical vectors",type=str)
    parser.add_argument("-temp", "-template", dest="template",
        help="Template structure used for finding important amino acids, must only contain protein atoms",type=str)
    parser.add_argument("-msa", "-alignment", dest="MSA",
        help="The input MSA used for matching and extracting important residues",type=str)
    parser.add_argument("-name", "-template_name", dest="name",
        help="Name/identifier of template enzyme in MSA",type=str)
    parser.add_argument("-ir", "-important_residues", dest="IR",type=int, nargs='+',
        help="IDs of the manually selected important residues with respect to the template structure")
    parser.add_argument("-cen", "-centroids", dest="centroid",type=int, nargs='+',
        help="IDs of residues used as centroids in sphere extraction with respect to the template structure")
    parser.add_argument("-rad", "-radius", dest='rad',
        help="Radius of sphere in nanometers when using using the sphere extraction method",type=float)
    parser.add_argument("-o", "-out", dest='out',
        help="File name of the output protein representation",type=str)
    
    return parser

def sphere_extraction(template_struc,rad,centroids):
    '''
    Creates spheres using given residues and centroids and extracts all residues within
    :param template_struc: template structure
    :param rad: radius of sphere
    :param centroids: list of centroids
    :return: residue ids and residue names+ids of important residues
    '''

    # Initialize top
    top = template_struc.topology
    table, _ = top.to_dataframe()

    # Extract sphere indices from structure
    sph_idx = None
    for cen in centroids:
        centroid_id = top.select(f'(resid {cen}) and (name == CA)')
        sph_idx = md.compute_neighbors(template_struc,rad,centroid_id)[0] if sph_idx is None else sph_idx+md.compute_neighbors(template_struc,rad,centroid_id)[0]
    sph_idx = np.unique(sph_idx)

    # Convert indices to residue IDs and names
    neighbors = table[table['serial'].isin(sph_idx)]
    neighbors_noH = neighbors[neighbors.element!='H']
    sph_resid = np.unique(neighbors_noH.resSeq)
    sph_res = []
    for id in sph_resid:
        sph_res.append(np.unique(table[table.resSeq==id].resName)[0]+f'{id}')
    
    return sph_resid, sph_res

def manual_selection(template_struc, IR):
    '''
    Creates array of residue ids and list of residue names+ids of important residues.
    :param template_struc: template structure
    :param IR: List of IDs of important residues
    :return: residue ids and residue names+ids of important residues
    '''
    
    # Initialize top
    top = template_struc.topology
    table, _ = top.to_dataframe()

    # Convert IDs to residue names
    IR_res = []
    for id in IR:
        IR_res.append(np.unique(table[table.resSeq==id].resName)[0]+f'{id}')

    return np.array(IR), IR_res
    
def matching(align,name,resid,res):
    '''
    Matching all proteins in alignment with important residues from template protein
    :param align: MSA
    :param name: Name of template protein
    :param resid: List of IDs for important residues
    :param res: List of important residues
    :return: DataFrame of matched residues for all proteins in MSA
    '''
    
    # Define all protein names and template protein alignment
    prot_names = []
    for i,r in enumerate(align):
        prot_names.append(r.id)
        if r.id == name:
            temp_idx = i
    template_align = align[temp_idx].seq
    
    # Identify indices of important residues in template protein alignment
    res_id = 1
    align_idx = []
    for i,aa in enumerate(template_align):
        if aa != '-':
            if np.any(resid==res_id):
                align_idx.append(i)
            res_id+=1
    # Match MSA with important residues 
    matched = np.empty((len(prot_names),len(align_idx)),dtype='str')
    for i,idx in enumerate(align_idx):
        matched[:,i] = np.array(list((align[:,idx])))
    matched_df = pd.DataFrame(matched,columns=res,index=prot_names)

    # Remove duplicate rows with identical indices
    matched_df = matched_df.groupby(matched_df.index).first()

    return matched_df

def encoding_matched(matched,encoding):
    '''
    Encode all proteins using their matched residues and the given sequence encoding
    :param matched: DataFrame of matched residues
    :param encoding: Sequence encoding
    :return: DataFrame of encoded proteins
    '''
    
    # Make column list to name features by combining name of important residues and sequence encoding identifiers
    cols = [name+' '+prop for name, prop in product(matched.columns.to_list(),encoding.index)]

    # Encode all proteins using the given sequence encoding
    encoded = []
    for i,enz in enumerate(matched.index):
        encoded.append(encoding[matched.iloc[i].to_numpy()].to_numpy().flatten())
    encoded = pd.DataFrame(encoded,index=matched.index,columns=cols)

    # Rename the index for use in modular architecture
    encoded.index.names = ['enzyme']

    return encoded

def main(args):
    # Initialize variables
    print('Initializing')
    template_struc = md.load_pdb(args.template)
    align = AlignIO.read(args.MSA, "fasta")
    encoding = pd.read_csv(args.encode,index_col=0)

    print('Identifying residues')
    # If important residues are manually specified, use these
    if args.IR:
        print('Using manual selection')
        resid, res = manual_selection(template_struc, args.IR)
    # If only centroids are specified, run sphere extraction
    elif args.centroid and args.IR is None:
        print('Using sphere extraction')
        resid, res = sphere_extraction(template_struc,args.rad,args.centroid)
    # Manual selection has priority over sphere extraction
    elif args.centroid and args.IR:
        print('Both centroids and manually selected residues are specified, using manual selection of important residues')
        resid, res = manual_selection(template_struc, args.IR)
    else:
        print('Neither centroids nor manual selection specified, unable to run encoding')
        return

    # Matching
    print('Matching')
    matched = matching(align,args.name,resid,res)

    # Encoding
    print('Encoding')
    encoded = encoding_matched(matched,encoding)

    encoded.to_csv(args.out,sep='\t')


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)


