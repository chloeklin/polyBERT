#!/bin/bash

#PBS -q gpuvolta
#PBS -P p00
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=200GB
#PBS -l walltime=10:00:00
#PBS -l storage=scratch/p00+gdata/dk92
#PBS -l jobfs=100GB


module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.10.4 cuda/12.5.1 
module list

cd /scratch/p00/hl4138
source test-venv/bin/activate


cd polyBERT/polyBERT
python3 evaluate.py --size "1M" --batch 512
