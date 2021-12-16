<a href="https://doi.org/10.1101/2021.12.14.469105"><img alt="" src="https://img.shields.io/badge/bioRχiv%20DOI-10.1101/2021.12.14.46910-blue?style=flat"/></a>

**Abstract**: Despite the importance of microbial dysbiosis in human disease, the phenomenon remains poorly understood. We provide the first comprehensive and predictive model of dysbiosis at ecosystem-scale, leveraging our new machine learning method for efficiently inferring compact and interpretable dynamical systems models. Coupling this approach with the most densely temporally sampled interventional study of the microbiome to date, using microbiota from healthy and dysbiotic human donors that we transplanted into mice subjected to antibiotic and dietary interventions, we demonstrate superior predictive performance of our method over state-of-the-art techniques. Moreover, we demonstrate that our approach uncovers intrinsic dynamical properties of dysbiosis driven by destabilizing competitive cycles, in contrast to stabilizing interaction chains in the healthy microbiome, which have implications for restoration of the microbiome to treat disease.

Important links
- Main Paper (Pre-print): ["Intrinsic instability of the dysbiotic microbiome revealed through dynamical systems inference at scale"](https://doi.org/10.1101/2021.12.14.469105)<br />
  <a href="https://doi.org/10.1101/2021.12.14.469105"><img alt="" src="https://img.shields.io/badge/bioRχiv%20DOI-10.1101/2021.12.14.46910-blue?style=flat"/></a>
- Associated GitHub repo for the ML model: ["MDSINE2"](https://github.com/gerberlab/MDSINE2)<br />
  <a href="https://github.com/gerberlab/MDSINE2"><img alt="" src="https://img.shields.io/badge/GitHub-MDSINE2-blue?style=flat&logo=github"/></a>
- Folder containing [tutorials as notebooks exploring the model, data and paper](https://github.com/gerberlab/MDSINE2_Paper/tree/master/google_colab) that can be opened directly in Google Colab<br />
<a href="https://github.com/gerberlab/MDSINE2_Paper/tree/master/google_colab"><img alt="" src="https://img.shields.io/badge/Jupyter Notebooks-MDSINE2%20Tutorials-blue?style=flat&logo=jupyter"/></a>


### References
Pre-print
```bibtex
@article {Gibson2021.12.14.469105,
	author = {Gibson, Travis E and Kim, Younhun and Acharya, Sawal and Kaplan, David E and DiBenedetto, Nicholas and Lavin, Richard and Berger, Bonnie and Allegretti, Jessica R and Bry, Lynn and Gerber, Georg K},
	title = {Intrinsic instability of the dysbiotic microbiome revealed through dynamical systems inference at scale},
	year = {2021},
	doi = {10.1101/2021.12.14.469105},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/12/16/2021.12.14.469105},
	journal = {bioRxiv}}
```


## Analysis in the cloud
We have provided interactive notebooks for parsing of raw and pre-preprocessed data, performing inference with our model, and reproducing the manuscript figures from pickle files containing posterior samples from full inference runs. Simply go to [google_colab/](google_colab/) and the notebooks can be launched directly into Google Colab

## Analysis on a local machine

This section outlines how to run MDSINE2 analysis on our dataset in full, with `bash`, `conda` and `git`.

### Setup

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

Once the above installation done, one can run a local copy of the jupyter notebooks found in [google_colab/](google_colab/).
```
conda install -c conda-forge jupyterlab
jupyter-notebook
```
Navigate to `google_colab/` to access the notebooks.


### Full Run

The inference performed in the jupyter notebooks are miniature versions (so as to execute in a reasonable amount of time). 
For the full run, assuming that the MDSINE2 core package is installed, follow the instructions located 
in [the analysis subfolder](analysis/README.md).
