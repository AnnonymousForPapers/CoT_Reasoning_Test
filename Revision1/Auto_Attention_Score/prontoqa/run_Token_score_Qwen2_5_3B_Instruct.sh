#!/bin/bash

#SBATCH --partition=general-gpu               # Name of Partition
#SBATCH --gres=gpu:1                          # Request 1 GPU cards for the job
#SBATCH --constraint=epyc64
#SBATCH --constraint=a100
#SBATCH --output=R-%x_%j.out

module purge

source /gpfs/homefs1/ych22001/miniconda3/etc/profile.d/conda.sh

conda activate HF_LLM

python run_attention_analysis.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path logic_proving_dataset.csv
