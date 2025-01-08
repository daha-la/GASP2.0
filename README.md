# GASP2.0
An updated version of the original Glycosyltransferase Acceptor Specificity Predictor (GASP) model. In this new version, the encoding pipelines and model architecture has been revamped to allow for enhanced processing of the enzyme infomation.

## Installation
To install GASP2.0:
- Make sure Anaconda or Miniconda is installed
- Clone this GitHub repository
- Create the two following conda environments (requirement files can be found in [src/environments](src/environments/)) using the commands below
```bash
conda create -n GASP2.0 python=3.11
conda activate GASP2.0
conda install pip
pip install -r GASP2_requirements.txt
```
```bash
conda create -n prog python=3.12
conda activate prog
conda install pip
pip install -r prog_requirements.txt
```

To run the original version of GASP, please also create a third environment
```bash
conda create -n GASP1.0 python=3.10
conda activate GASP1.0
conda install pip
pip install -r GASP1_requirements.txt
```

## Run
All scripts for running GASP2.0 can be found in the [RUN](RUN) subfolder. Simply run the bash scripts in the terminal.
For example, to run the default random forest version of GASP2.0, use the following command:
```bash
bash RUN_rf.sh
```
This will first use the training dataset in [Data](Data) to create a random forest model called "model_blosum62Amb_Nterm_default" in the [Results/models](Results/models) folder. Secondly, it will employ this model to predict the acceptor specificity of the GT1:acceptor pairs found in the test dataset (also in [Data](Data)). This prediction output is saved in [Results/predictions](Results/predictions) as "pred_blosum62Amb_Nterm_default.tsv".

To change the protein encoding strategy, please edit the "encoding" variable in the bash script. To use alternative datasets for training or testing, please edit the corresponding variables. Remember to also edit the "identifier" variable to ensure no prior predictions are overwritten.

## Encodings
New encodings can be created using the notebooks in the [Representation_Strategies](Representation_strategies) subfolders.
For example, to create a new alignment based protein representation, navigate to [Representation_Strategies/protein/methods/Alignment](Representation_Strategies/protein/methods/Alignment) and use the "alignment_embedding.ipynb". The resulting encoding can be found in the [Representation_Strategies/protein/encodings](Representation_Strategies/protein/encodings).

## New Predictions
When making predictions or training alternative models on new datasets, do the following steps:
- Save the new datasets in the [Data](Data) folder
- Create a new fasta file in [Data/fasta](Data/fasta) containing all proteins used for training and testing.
- Obtain the representations of the proteins from the new fasta file using the encoding techniques found in [Representation_Strategies/protein/methods](Representation_strategies/protein/methods/). Remember to change the variables of the notebooks to match the fasta file. When using "progres", remember to create new subfolders for the 3D structures of any new protein.
- Run the desired architecture using the bash scripts found in [Run](Run), changing the variables to fit the new files and encodings. The new predictions can be found in [Results/predictions](Results/predictions)

## Attribution  
GASP2.0 is an extension of the original GASP model, developed by David Harding-Larsen and Christian Degnbol Madsen.  
The original GASP implementation can be found at [https://github.com/degnbol/GASP](https://github.com/degnbol/GASP).  

GASP2.0 builds upon this foundation with additional features, improvements, and modifications introduced by the current development team.

## License  
This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.  
See the [LICENSE](LICENSE) file for details.
