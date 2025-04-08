#!/bin/bash

#PBS -q hugemembw
#PBS -P hm62
#PBS -l ncpus=140
#PBS -l mem=1280GB
#PBS -l walltime=05:00:00
#PBS -l storage=scratch/hm62+gdata/dk92


module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.10.4 cuda/12.5.1 gcc/14.1.0 deepspeed/0.15.1
cd /scratch/hm62/hl4138
source test-venv/bin/activate
cd polyBERT/polyBERT
python3 evaluate_ds.py --size "1M" --ngpus 140



