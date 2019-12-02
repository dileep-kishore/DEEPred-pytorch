#!/usr/bin/env bash
#$ -P visant
#$ -pe omp 4
#$ -o data/model_outputs/$1
#$ -e data/models_outputs/

source activate deepred_pytorch

OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
python train_1_model.py $1