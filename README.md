# MDSINE2: Microbial Dynamical Systems Inference Engine 2

****************************************High level description of the package

## Installation

## Tutorials
For tutorials on running MDSINE2, post-processing, and how to use the MDSINE2 software, see `tutorials`.

## Quick start
#### Running MDSINE2 with Gibson dataset
The raw data of the Gibson dataset is in the folder `datasets/gibson`. To run the MDSINE2 model, use the scripts in the the folder `gibson_dataset`. 

#### Running MDSINE2 with your own dataset
*****************************************************Describe the required tables

1) Parse the tables into a `Study` object
   First we need to parse in the tables of raw data into an MDSINE2 `Study` object
    ```bash
    python step-1_parse_data.py \
        --name name-of-dataset \
        --taxonomy path/to/taxonomy-table.tsv \
        --metadata path/to/metadata-table.tsv \
        --reads path/to/reads-table.tsv \
        --qpcr path/to/qpcr-table.tsv \
        --outfile output/study.pkl
    ```
2) (Optional) Filter the data
    Filter out taxas that don't have enough information
    ```bash
    python step_2_filtering.py \
        --dataset output/study.pkl \
        --outfile output/study_filtered.pkl \
        --dtype DTYPE \
        --threshold THRESHOLD \
        --min-num-consecutive MIN_NUM_CONSECUTIVE \
        --min-num-subjects MIN_NUM_SUBJECTS \
        --colonization-time COLONIZATION_TIME
    ```

3) Learn parameters of MDSINE2 model
   Run the inference of MDSINE2
   ```bash
   python step_5_infer_mdsine2.py \
       --input output/study_filtered.pkl \
       --negbin A0 A1 \
       --seed SEED \
       --burnin BURNIN \
       --n-samples N_SAMPLES \
       --checkpoint CHECKPOINT \
       --basepath output \
       --multiprocessing 0
   ```

4) Visualize the MDSINE2 parameters
   Visualize the parameters
   ```bash
   python step_6_visualize_mdsine2.py \
       --chain output/name-of-dataset/mcmc.pkl \
       --output-basepath output/name-of-dataset/posterior \
       --section posterior
    ```



