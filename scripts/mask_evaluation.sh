#!/bin/bash

#PBS -l ncpus=7
#PBS -l mem=32GB
#PBS -q normalbw
#PBS -P p00
#PBS -l walltime=02:00:00
#PBS -l storage=scratch/hm62

cd /scratch/p00/hl4138

module load python3/3.10.4

source test-venv/bin/activate
cd polyBERT/polyBERT

python3 mask_evaluation.py --size "50M"

#1hr for 1M and 5M