# Install dependencies (Python 3.7.3)
* [mdsine2](https://github.com/gerberlab/PyLab) == 4.0.1
* biopython==1.76
* scikit-bio==0.5.6
* ete3

# Running the model
```python
import mdsine2 as md2
```
Run the MDSINE2 model with the Gibson dataset


#### Fit the Negative Binomial Dispersion parameters
```python
# Get the replicate data
subjset = md2.dataset.gibson(dset='replicate')
# Run the model
TODO
```

#### Fit the qPCR measurements
```python
```

#### Run the Model

These commands run and save the model, plots the posteriors, and runs validation (if possible)

```python
# Healthy cohort
```
```python
# Ulcerative Colitis Cohort
```
```python
# Leave one first subject of Ulcerative Colitis for validation
```
```python
# Dispatch new lsf job for leave one out validation
```