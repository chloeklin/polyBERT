#!/bin/bash

#PBS -l ncpus=7
#PBS -l mem=20GB
#PBS -q normalbw
#PBS -P um09
#PBS -l walltime=00:30:00
#PBS -l storage=scratch/um09
#PBS -l jobfs=1GB

module load python3/3.10.4
cd /scratch/um09/hl4138
source polybert-venv/bin/activate


cd polyBERT
unzip polyBERT.zip -x "__MACOSX/*" "*.DS_Store"
mv -v polyBERT/ $PBS_JOBFS

python3 tokeniser/do_tokenize.py --root_dir $PBS_JOBF/polyBERT --size "50M"

cd $PBS_JOBFS
zip -r polyBERT.zip polyBERT -x "*/.ipynb_checkpoints/*"
mv -v polyBERT.zip /scratch/um09/hl4138/polyBERT/