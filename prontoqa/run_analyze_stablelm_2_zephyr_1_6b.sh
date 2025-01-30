#!/bin/bash

#SBATCH --partition=general               # Name of Partition
#SBATCH --output=R-%x_%j.out

module purge

source /gpfs/homefs1/ych22001/miniconda3/etc/profile.d/conda.sh

conda activate HF_LLM

python analyze_results.py stablelm-2-zephyr-1_6b_proofbycontra_1hop_ProofByContra_nodistractor.log