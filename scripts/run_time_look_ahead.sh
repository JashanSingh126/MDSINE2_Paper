#!/bin/bash
cd ../time_look_ahead
python dispatch.py --queue short --input input/basepaths.txt --n-cpus 1 --n-mbs 4000 --n-days 8 --run-jobs 1