#!/bin/bash

#PBS -l ncpus=7
#PBS -l mem=32GB
#PBS -q normalbw
#PBS -P hm62
#PBS -l walltime=00:05:00
#PBS -l storage=scratch/hm62



cd /scratch/hm62/hl4138
source test-venv/bin/activate
module load python3/3.10.4 
cd polyBERT/polyBERT
python3 tokenise_evaluate.py --size "1M" 




