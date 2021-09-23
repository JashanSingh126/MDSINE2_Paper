# MDSINE2 Inference on Mouse Experiment Dataset

(Last updated: 2021 Sept. 23)

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

First, one needs to run DADA2 (https://benjjneb.github.io/dada2/index.html) on the 16S reads, to end up with a list of ASVs and taxonomic assignments 
using RDP (`rdp_species.tsv`) and Silva (`silva_species.tsv`).
To do this, assuming DADA2 is installed, run the following script:

```
TODO
```

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

(Under construction. Enrichment/cross-validation errors/keystoneness/cycle counting/)
