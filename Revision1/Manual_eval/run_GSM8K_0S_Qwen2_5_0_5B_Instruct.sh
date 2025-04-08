#!/bin/bash

#SBATCH --partition=general-gpu               # Name of Partition
#SBATCH --gres=gpu:1                          # Request 1 GPU cards for the job
#SBATCH --constraint=a100
#SBATCH --mem=256G                            # Request 256GB of available RAM
#SBATCH --output=R-%x_%j.out

module purge

source /gpfs/homefs1/ych22001/miniconda3/etc/profile.d/conda.sh

conda activate HF_LLM

python GSM8K_0_shots.py \
  --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
  --shots 0
