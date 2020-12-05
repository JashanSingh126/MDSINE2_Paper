# ErisOne documentation [Internal use]

### WARNING
__These are for internal use only__. If you try to run these scripts, the job will fail. These jobs run the same as the jobs in the parent folder - they just are submitting as jobs using LSF.

### Forward simulation and cross-validation
To run cross validation and forward simulation, run
```bash
./submit_cv_and_tla.sh
```
Note that computing forward simulation submits many jobs to the short and medium queue

### Run the model
Run the model with two different seeds
```bash
./submit_infer_mdsine2_and_fixed_top.sh
```