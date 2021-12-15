# Intrinsic instability of the dysbiotic microbiome revealed through dynamical systems inference at scale
<a href="https://"><img alt="" src="https://img.shields.io/badge/DOI-addmelater-blue?style=flat"/></a>

**Abstract**: Despite the importance of microbial dysbiosis in human disease, the phenomenon remains poorly understood. We provide the first comprehensive and predictive model of dysbiosis at ecosystem-scale, leveraging our new machine learning method for efficiently inferring compact and interpretable dynamical systems models. Coupling this approach with the most densely temporally sampled interventional study of the microbiome to date, using microbiota from healthy and dysbiotic human donors that we transplanted into mice subjected to antibiotic and dietary interventions, we demonstrate superior predictive performance of our method over state-of-the-art techniques. Moreover, we demonstrate that our approach uncovers intrinsic dynamical properties of dysbiosis driven by destabilizing competitive cycles, in contrast to stabilizing interaction chains in the healthy microbiome, which have implications for restoration of the microbiome to treat disease.

Important links
- (Pre-print): ["Intrinsic instability of the dysbiotic microbiome revealed through dynamical systems inference at scale"]()<br />
  <a href="https://"><img alt="" src="https://img.shields.io/badge/DOI-addmelater-blue?style=flat"/></a>
- Folder containing [Google Colab tutorials exploring the model, data and paper](https://github.com/gerberlab/MDSINE2_Paper/tree/master/google_colab) or open directly in colab using button below<br /><a href="https://colab.research.google.com/github/gerberlab/MDSINE2_Paper/blob/master/">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- Associated GitHub repo for MDSINE2 <br />
  <a href="https://github.com/gerberlab/MDSINE2"><img alt="" src="https://img.shields.io/badge/GitHub-MDSINE2-blue?style=flat&logo=github"/></a>

References
```
@InProceedings{pmlr-v80-gibson18a,
  title = 	 {Robust and Scalable Models of Microbiome Dynamics},
  author =       {Gibson, Travis and Gerber, Georg},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {1763--1772},
  year = 	 {2018},
  editor = 	 {Dy, Jennifer and Krause, Andreas},
  volume = 	 {80},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {10--15 Jul},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v80/gibson18a.html},
}
```





1. [Analysis in the cloud](#Cloud)  
2. [Analysis on a local machine](#Local)
    1. [Setup](#LocalSetup)
    2. [Jupyter Notebook - Short Run](#LocalJupyter)
    3. [Local Scripts - Full Run](#LocalFullRun)

<a name="Cloud"/>

## 1. Analysis in the cloud
Follow the link to an interactive run through of our analysis pipeline using MDSINE2, hosted on Google colab.

<a href="https://colab.research.google.com/github/gerberlab/MDSINE2_Paper/blob/master"><img alt="" src="https://img.shields.io/static/v1?label=Colab&message=Launch%20in%20Google%20Colab&color=orange&logo=googlecolab&style=for-the-badge&logoWidth=10"/></a>

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

Once the above installation done, one can run a local copy of the jupyter notebooks found in [Analysis in the cloud](#Cloud).
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
