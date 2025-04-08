#!/bin/bash

#PBS -q gpuvolta
#PBS -P hm62
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=382GB
#PBS -l walltime=010:00:00
#PBS -l storage=scratch/hm62+gdata/dk92
#PBS -l jobfs=50GB

cd /scratch/hm62/hl4138
source test-venv/bin/activate

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.10.4 cuda/12.5.1 gcc/14.1.0 deepspeed/0.15.1
module list

cd polyBERT/polyBERT
python3 evaluate.py --size "1M" 




