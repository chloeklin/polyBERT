#!/bin/bash

#PBS -l ncpus=12
#PBS -l ngpus=3
#PBS -l mem=32GB
#PBS -q gpuvolta
#PBS -P hm62
#PBS -l walltime=06:00:00
#PBS -l storage=scratch/hm62

module load python3/3.9.2
cd polyBERT
python3 train_tokenizer.py --size "1M"
python3 do_tokenize.py --size "1M"
python3 pretrain.py




