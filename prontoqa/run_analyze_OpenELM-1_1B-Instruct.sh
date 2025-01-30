#!/bin/bash

#SBATCH --partition=general               # Name of Partition
#SBATCH --output=R-%x_%j.out

module purge

source /gpfs/homefs1/ych22001/miniconda3/etc/profile.d/conda.sh

conda activate HF_LLM

python analyze_results.py OpenELM-1_1B-Instruct_andelim_1hop_AndElim_nodistractor.log