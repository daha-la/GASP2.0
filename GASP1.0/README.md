# GASP1.0
This folder contains all code nescessary for running the first version of GASP. For more information, please refer to the [GASP paper](https://pubs.acs.org/doi/10.1021/acsomega.4c01583) and the [GitHub](https://github.com/degnbol/GASP).

Before running any scripts, please activate the conda environment for GASP1.0.
````bash
conda activate GASP1.0
````
To run the model with the original parameters and datasets, please activate the GASP1.0 conda environment and execute the bash script
````bash
bash RUN_GASP1.0.sh
````
When running an alternative testset - such as when evaluting the predicted acceptor specificity for multiple protein on the same acceptor, please use the bash script for running GASP using new data. This data should be saved in the [Data](../Data/) folder.
````bash
bash RUN_GASP1.0_newdata.sh "new_data" original
````
Please change the "new_data" to the name of the dataset. If the dataset is a combination dataset, please change the optional variable. For more information, see the [Run](../Run/) folder.
````bash
bash RUN_GASP1.0_newdata.sh "new_data" original yes
````
For evalutating new proteins, please first create a new fasta file containing the new proteins (in .faa format) and save it in the [fasta](../Data/fasta/) folder. As for the running GASP with new data, please save this new dataset in the [Data](../Data/) folder. Finally, run the encoding script for new sequences use the GASP bash script for new data.
````bash
bash encode_new_seqs_GASP.sh "new_seqs"
bash RUN_GASP1.0_newdata.sh "new_data" "new_seqs"
````
Please change the "new_seqs" to the name of the fasta file and "new_data" to the name of the new dataset when executing the scripts. Again, if the dataset is a combination dataset, please change the optional variable as above for the run script.
