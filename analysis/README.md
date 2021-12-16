# MDSINE2 Inference on Mouse Experiment Dataset

(Last updated: 2021 Dec. 16)

To run the scripts in this README, go into the `analysis` directory.

```
cd analysis
```

# 1. Preprocess the Data

We provide pre-processed outputs for this section in the repository.

- The DADA2 outputs are in `../datasets/gibson/`, along with the other input TSV files necessary for inference.
- The ASV-to-OTU aggregation outputs are in `./output/gibson/preprocessed`.
- Phylogenetic placement of OTUs are in `./files/phylogenetic_placement_OTUs`.

The following subsections can be run to re-generate these outputs from scratch.

## 1.1 Run DADA2

If starting with the raw 16S reads (fasta files), one needs to run DADA2 (https://benjjneb.github.io/dada2/index.html), 
to end up with a list of ASVs and taxonomic assignments using RDP (`rdp_species.tsv`) and 
Silva (`silva_species.tsv`).

We have provided the R script used to perform this procedure, located at (relative to this directory) `./dada/dada2_script.R`.

## 1.2 Aggregate ASVs into OTUs

This step implements the creation of OTUs from DADA2's ASV output, as outlined in
"Methods - ASV aggregation into OTUs".
```
bash gibson_inference/preprocessing/preprocessing_agglomeration.sh
```
This script automatically parses the input TSV files containing counts, subject metadata, 
perturbation windows, qpcr and the DADA2 outputs -- they are located in `../datasets/gibson/`.

## 1.3 Assign OTU Taxonomy

This step implements the taxonomic assignment of OTUs found in "Methods - ASV aggregation into OTUs".
```
bash gibson_inference/preprocessing/assign_consensus_taxonomy.sh
```

## 1.4 Filter OTUs from Input

Next, filter out OTUs which don't colonize consistently across the subjects, as explained in 
"Methods - Filtering to Remove Low Abundance Taxa".
```
bash gibson_inference/preprocessing/preprocessing_filtering.sh
```

## Optional 1: Plot the empirical abundances of OTUs and constituent ASVs.

To see how the OTUs' measured abundances compare to their constituent ASVs, generate the time-series plots using raw counts and qPCR:

```
bash gibson_inference/preprocessing/plot_aggregates.sh
```

## Optional 2: Visualize phylogenetic subtrees containing each OTU.

This implements "Methods - Phylogenetic placement of sequences", which performs a multiple alignment and then places each
sequence at the leaves of a tree:

```
TODO
```

We visualize these by using the following command:
```
bash gibson_inference/preprocessing/plot_phylogenetic_subtrees.sh
```

# 2. Learn Negative Binomial Dispersion Parameters 

Learn the negative binomial dispersion parameters (d0 and d1) from the error model of our paper 
(See Supplemental Methods for the mathematical description).
```
bash gibson_inference/inference/learn_negbin.sh
```

# 3. Run MDSINE2's inference

Now, we run the following script, which uses the results of the previous steps to perform the MCMC algorithm described
in our paper.
```
bash gibson_inference/inference/run_mdsine2.sh
```

Next, using the previous MCMC run as input, we perform the "fixed-clustering" run.
This performs inference using a "consensus-clustering" as described in "Methods - Consensus Modules".
```
bash gibson_inference/inference/run_mdsine2_fixed_clustering.sh
```

# 4. Downstream Analyses

## 4.1. Enrichment Analysis

Sawal TODO

## 4.2 Simulated-based Stability Analysis

First, run the necessary forward simulations. This script runs many forward simulations for a variety of randomly
sampled conditions, and thus takes some time. It is not parallelized across the configurations, 
though we ran a parallelization script (not included here) for our private compute servers.

```
bash gibson_inference/downstream_analysis/stability/evaluate_stability_simulated.sh
```

The results of the forward simulations (the steady states) are collectively compiled and saved to dataframes.
Render the plots using the following script:

```
bash gibson_inference/downstream_analysis/stability/plot_stability.sh
```

## 4.3 Eigenvalues

Plot a histogram of the positive real parts of eigenvalues across all the MCMC samples:

```
bash gibson_inference/downstream_analysis/eigenvalues/plot_eigenvalues.sh
```

## 4.4 Cycle counting

Plot the counts of all module-to-module cycles of fixed MDSINE2 runs for Healthy and Dysbiotic cohorts:

```
bash gibson_inference/downstream_analysis/cycles/plot_cycles.sh
```

## 4.5 Keystoneness

First, run the necessary forward simulations. For each choice of module from a fixed-cluster MDSINE2 run,
initialize it to zero (all other OTUs are initialized to their respective day-20 levels).

```
bash gibson_inference/downstream_analysis/keystoneness/evaluate_keystoneness.sh
```

The results of the forward simulations (the steady states) are collectively compiled and saved to dataframes.
Render the plots using the following script:

```
bash gibson_inference/downstream_analysis/keystoneness/plot_keystoneness.sh
```
