#!/usr/bin/env bash
#$ -P visant
#$ -pe omp 8

source activate deepred_pytorch

OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
python -W ignore train_1_model.py $1
