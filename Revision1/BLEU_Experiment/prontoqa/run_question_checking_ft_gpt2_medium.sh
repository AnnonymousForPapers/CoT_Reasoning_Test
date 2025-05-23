#!/bin/bash

#SBATCH --partition=general-gpu               # Name of Partition
#SBATCH --gres=gpu:1                          # Request 1 GPU cards for the job
#SBATCH --constraint=epyc64
#SBATCH --constraint=a100
#SBATCH --output=R-%x_%j.out

module purge

source /gpfs/homefs1/ych22001/miniconda3/etc/profile.d/conda.sh

conda activate HF_LLM

python question_checking.py --model_name gpt2-medium --model_path "/home/ych22001/LLM/Reasoning_ability/prontoqa/gpt2-medium-finetuned-Last2-prontoqa/checkpoint-18676"
