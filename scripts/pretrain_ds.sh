#!/bin/bash

#PBS -q gpuvolta
#PBS -P um09
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=100GB
#PBS -l walltime=01:00:00
#PBS -l storage=scratch/um09+gdata/dk92
#PBS -l jobfs=100GB


cd /scratch/um09/hl4138
source polybert-venv/bin/activate

module use /g/data/dk92/apps/Modules/modulefiles
module load python3/3.10.4 cuda/12.5.1 gcc/14.1.0 deepspeed/0.15.1 NCI-ai-ml/24.11
module list

cd polyBERT
unzip polyBERT.zip -x "__MACOSX/*" "*.DS_Store"
mv -v polyBERT/ $PBS_JOBFS

python3 model/pretrain_ds.py --root_dir $PBS_JOBFS/polyBERT/ --size "1M" --ngpus 4


zip -r $PBS_JOBFS/polyBERT.zip $PBS_JOBFS/polyBERT
mv -v $PBS_JOBFS/polyBERT.zip ./

 



