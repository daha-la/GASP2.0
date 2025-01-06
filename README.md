# GASP2.0
An updated version of the original Glycosyltransferase Acceptor Specificity Predictor (GASP) model. In this new version, the encoding pipelines and model architecture has been revamped to allow for enhanced processing of the enzyme infomation.

## Installation
To install GASP2.0:
- Make sure Anaconda or Miniconda is installed
- Copy this GitHub repository
- Create the conda environments (requirement files can be found in [src](src)) using the commands below
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


## Attribution  
GASP2.0 is an extension of the original GASP model, developed by David Harding-Larsen and Christian Degnbol Madsen.  
The original GASP implementation can be found at [https://github.com/degnbol/GASP](https://github.com/degnbol/GASP).  

GASP2.0 builds upon this foundation with additional features, improvements, and modifications introduced by the current development team.

## License  
This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.  
See the [LICENSE](LICENSE) file for details.
