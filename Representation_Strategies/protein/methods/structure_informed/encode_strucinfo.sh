#!/usr/bin/env zsh
sequence_encoding="../encodings/raw/atchley.csv"
sequence_encoding="../encodings/raw/blosum62Amb_alt.csv"
template_structure="../data/6SU6.pdb"
MSA="../alignment/seqs.afa"
template_name='Pt_UGT1'
#important_residues=" 10 13 14 21 22 48 50 55 64 66 68 75 81 86 91 94 98 107 110 112 117 119 121 122 125 126 129 132 137 139 140 146 147 148 157 173 176 181 186 188 190 193 196 198 208 210 222 224 227 235 238 259 264 267 269 271 273 274 278 279 285 286 290 296 297 299 300 307 308 332 333 336 337 340 341 342 343 346 351 363 365 368 381 387 388 390 395 396 399 405 409 412 413 424 430 455 457 460 462 463 472"
important_residues=" 198 399 75 381 222 296 430 188 148 190 86 110 413 146 388 297 "
centroids=" 26 "
radius=2.5


outfile="../encodings/h26_25sph_blosum.tsv"
#outfile="../encodings/Pt_RRIR.tsv"

# Manual
#python StuctureInformed.py -enc $sequence_encoding -temp $template_structure -msa $MSA -name $template_name -ir $important_residues -rad $radius -out $outfile

# Sphere extraction
python StuctureInformed.py -enc $sequence_encoding -temp $template_structure -msa $MSA -name $template_name -cen $centroids -rad $radius -out $outfile