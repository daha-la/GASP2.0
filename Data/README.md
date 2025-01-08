# Data
This folder contains all the data used for training and testing GASP2.0 models.
When using alternative datasets, please save them here to ensure compatability with all pipelines. These datasets should be saved in the tsv format using tab seperators, containing three columns: one for enzymes names, one for binary activity, and one for substrate specifier (currently only CID is supported). Please ensure that the enzyme names match the names in the fasta file.
When using template-based representations, please ensure that the template pdb is saved in this folder.
## fasta
This folder contains the fasta files used for creating protein representations. For new fasta files, please ensure that all proteins used for training and testing is contained in a single fasta file.