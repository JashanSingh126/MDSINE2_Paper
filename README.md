# MDSINE2: Microbial Dynamical Systems Inference Engine 2

This repository contains the analysis pipeline found in our paper, with tutorials on how to use it.

1. [Analysis in the cloud (Google colab)](#Cloud)  
2. [Analysis on a local machine](#Local)
    1. [Setup](#LocalSetup)
    2. [Script - Project Structure](#LocalRun)

<a name="Cloud"/>

## 1. Analysis in the cloud (Google colab)
Follow the link to an interactive run through of our analysis pipeline using MDSINE2, hosted on Google colab.
This analysis includes the parsing of the raw input and all of the pre-preprocessing steps found in the 
Methods section of our paper.
Note that to meet the memory and time budget, the number of taxa and number of MCMC iterations are reduced and does not
fully reproduce the results in the paper.
For the full version, refer to [Analysis on a local machine](#Local).


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gerberlab/MDSINE2_Paper/HEAD?filepath=bindertutorials)

(Colab link under construction)

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

Next, clone and install the core MDSINE2 package (MCMC implementation) from this repository.

```
git clone https://github.com/gerberlab/MDSINE2
pip install MDSINE2/.
```

Next, clone this repository which contains the data and scripts to perform the analysis.

```
git clone https://github.com/gerberlab/MDSINE2_Paper
```

<a name="LocalRun"/>

### 2.1 Shell Scripts

Instructions for performing analysis is located in [the analysis subfolder](analysis/README.md).
