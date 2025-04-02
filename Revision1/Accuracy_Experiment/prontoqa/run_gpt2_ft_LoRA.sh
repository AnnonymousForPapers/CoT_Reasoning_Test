#!/bin/bash

#SBATCH --partition=general-gpu               # Name of Partition
#SBATCH --gres=gpu:1                          # Request 1 GPU card for the job
#SBATCH --constraint=epyc64
#SBATCH --constraint=a100
#SBATCH --output=R-%x_%j.out

module purge

source /gpfs/homefs1/ych22001/miniconda3/etc/profile.d/conda.sh

conda activate HF_LLM

python gpt2_ft_LoRA.py