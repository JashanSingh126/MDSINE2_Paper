# Learning microbial dynamics at scale



This repository shows you how to use MDSINE2 (https://github.com/gerberlab/MDSINE2) with interactive Jupyter notebooks 
that can be run in the cloud or locally. It also contains all the scripts to reproduce the full analysis and figures from our 
paper (coming shortly).

1. [Analysis in the cloud](#Cloud)  
2. [Analysis on a local machine](#Local)
    1. [Setup](#LocalSetup)
    2. [Jupyter Notebook - Short Run](#LocalJupyter)
    3. [Local Scripts - Full Run](#LocalFullRun)

<a name="Cloud"/>

## 1. Analysis in the cloud
Follow the link to an interactive run through of our analysis pipeline using MDSINE2, hosted on binder.
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gerberlab/MDSINE2_Paper/HEAD?filepath=bindertutorials)

This analysis includes the parsing of the raw input and all of the pre-preprocessing steps found in the 
Methods section of our paper.
Note that to meet the memory and time budget, the number of taxa and number of MCMC iterations are reduced and does not
fully reproduce the results in the paper.
For the full version, refer to [Local Scripts - Full Run](#LocalFullRun).



<a name="Local"/>

## 2. Analysis on a local machine

This section outlines how to run MDSINE2 analysis on our dataset in full, with `bash`, `conda` and `git`.

<a name="LocalSetup"/>

### 2.1 Setup

One must first install the MDSINE2 package, according to the following instructions.
The recommended setup starts out by creating a new conda environment. 
The package was developed and tested on python 3.7.3.

```
conda create -n mdsine2 -c conda-forge python=3.7.3
conda activate mdsine2
```

Next, clone and install the core MDSINE2 package (MCMC implementation) from the package repository (https://github.com/gerberlab/MDSINE2).

```
git clone https://github.com/gerberlab/MDSINE2
pip install MDSINE2/.
```

Next, clone this repository which contains the data and scripts to perform the analysis.

```
git clone https://github.com/gerberlab/MDSINE2_Paper
cd MDSINE2_Paper
```

<a name="LocalJupyter"/>

### 2.2 Jupyter Notebook - Short Run

Once the above installation done, one can run a local copy of the jupyter notebooks.
```
conda install -c conda-forge jupyterlab
jupyter-notebook
```
Navigate to `bindertutorials/` to access the notebooks.


<a name="LocalFullRun"/>

### 2.3 Local Scripts - Full Run

The run coded into the jupyter notebooks are miniature versions which do not reproduce the results in the paper.
For the full run, assuming that the MDSINE2 core package is installed, follow the instructions located 
in [the analysis subfolder](analysis/README.md).
