# Replication of MDSINE2 results

### Preprocess the data
Before any inference, perform preprocessing:
```bash
./preprocessing_and_learn_negbin.sh
```

### Cross-validation and forward simulation
Order of scripts from start to finish of running forward simulation and cross validation:
```bash
./run_cv.sh
./run_tla.sh
./compute_errors_tla.sh
```

### Learning parameters of MDSINE2
Order of scripts from start to finish of generating the posteriors

```bash
./run_mdsine2.sh
./run_mdsine2_fixed_clustering.sh
```

### Making figures
Once cross-validation and learning the parameters are done, you can generate the figures used in the paper:
```bash
./make_figures.sh
```