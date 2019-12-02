#!/bin/bash -l
#$ -P visant
#$ -pe omp 4

source activate deepred_pytorch

python train_1_model.py
