# MDSINE2 Inference on Mouse Experiment Dataset

We assume that DADA2 has already been run on the 16S reads so that one has a list of ASVs with taxonomic assignments 
using RDP (`rdp_species.tsv`) and Silva (`silva_species.tsv`).
(We did this by running the script ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) TODO referendce DADA2 script here)

First, go into the `analysis` directory.

```
cd analysis
```

# 1. Preprocess the Data

## 1.1 Aggregate ASVs into OTUs

This step implements the creation of OTUs from DADA2's ASV output, as outlined in
"Methods - ASV aggregation into OTUs".
```
bash gibson_inference/preprocessing/preprocessing_agglomeration.sh
```
This script automatically points to the input TSV files containing counts, subject metadata, 
perturbation windows, qpcr and the DADA2 outputs -- they are located in `../datasets/gibson/`.

## 1.2 Assign OTU Taxonomy

This step implements the taxonomic assignment of OTUs found in "Methods - ASV aggregation into OTUs".
```
bash gibson_inference/preprocessing/assign_consensus_taxonomy.sh
```

## 1.3 Filter OTUs from Input

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
This performs inference using a "consensus-clustering" run as described in "Methods - Consensus Modules".
```
bash gibson_inference/inference/run_mdsine2_fixed_clustering.sh
```

# 4. Downstream Analyses

(Under construction. Enrichment/cross-validation errors/keystoneness/cycle counting/)
