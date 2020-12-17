# Test scripts

These are scripts that run shorter runs of everything to make sure that they work.

# Order to run the scripts
The order to run the scipts is the same as it is for normal.

### Preprocessing, plotting, and learning negative binomial
1. `preprocessing_agglomeration.sh`
2. `assign_consensus_taxonomy.sh`
3. `plot_aggregates.sh`
4. `plot_phylogenetic_subtrees.sh`
5. `preprocessing_filtering.sh`
6. `learn_negbin.sh`

### Run MDSINE2
7. `run_mdsine2.sh`
8. `run_mdsine2_fixed_clustering.sh`
9. `compute_keystoneness.sh`

### Run cross validation
10. `run_cv.sh`
11. `run_tla.sh`
12. `compute_errors_tla.sh`

### Make figures
13. `make_figures.sh`