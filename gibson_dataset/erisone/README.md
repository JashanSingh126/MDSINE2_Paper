# ErisOne documentation [Internal use]

### WARNING
__These are for internal use only__. If you try to run these scripts, the job will fail. These jobs run the same as the jobs in the parent folder - they just are submitting as jobs using LSF.

## Programatically generate LSF files for running jobs
Each bash script runs a Python script located in `scripts` that generates a LSF file with the desired parameters.

### Forward simulation and cross-validation
To run cross validation and forward simulation, run
```bash
./submit_cv_and_tla.sh
```
To only run forward simulation, run
```bash
./submit_tla.sh
```

### Run the model
Run the model and fixed_clustering with two different seeds
```bash
./submit_mdsine2.sh
```