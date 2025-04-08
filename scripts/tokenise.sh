#!/bin/bash

#PBS -l ncpus=7
#PBS -l mem=32GB
#PBS -q normalbw
#PBS -P um09
#PBS -l walltime=06:00:00
#PBS -l storage=scratch/um09

module load python3/3.10.4
cd /scratch/um09/hl4138
source polybert-venv/bin/activate


cd polyBERT
unzip polyBERT.zip -x "__MACOSX/*" "*.DS_Store"
mv -v polyBERT/ $PBS_JOBFS

python3 tokeniser/do_tokenize.py --root_dir $PBS_JOBFS/polyBERT/ --size "1M"




