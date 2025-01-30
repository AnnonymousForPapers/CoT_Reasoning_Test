#!/bin/bash

#SBATCH --partition=general               # Name of Partition
#SBATCH --output=R-%x_%j.out

module purge

source /gpfs/homefs1/ych22001/miniconda3/etc/profile.d/conda.sh

conda activate HF_LLM

python analyze_results.py gpt2_large_proofbycontra_1hop_ProofByContra_3shot_nodistractor.log