# Data
This folder contains all the data used for training and testing GASP2.0 models.

When using alternative datasets, please save them here to ensure compatability with all pipelines. These datasets should be saved in the tsv format using tab seperators, containing up to three columns: one for enzymes names, one for substrate specifier (currently only CID is supported), and potentially one for binary activity. Please ensure that the enzyme names match the names in the fasta file.

When using template-based representations, please ensure that the template pdb is saved in this folder.
## fasta
This folder contains the fasta files used for creating protein representations. For new fasta files, please ensure that all proteins used for training and testing is contained in a single fasta file.

## Combinations
The combinations.sh script allows for the creation of datasets containing all the combinations of all proteins from a fasta file and either all or a subset of the acceptors in the original acceptor file.
````bash
conda activate GASP2.0
bash combinations "fasta_name" "target acceptors"
````
Please change the "fasta_name" to the name of the fasta file and "target acceptors" to a list of all target acceptors in the format of "acceptor1 acceptor2 acceptor3 ..." etc. If use all acceptors in the acceptor file, please leave "target acceptors" blank.