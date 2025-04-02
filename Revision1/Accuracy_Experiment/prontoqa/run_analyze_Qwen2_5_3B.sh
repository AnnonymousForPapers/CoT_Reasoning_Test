#!/bin/bash

#SBATCH --partition=general               # Name of Partition
#SBATCH --output=R-%x_%j.out

module purge

source /gpfs/homefs1/ych22001/miniconda3/etc/profile.d/conda.sh

conda activate HF_LLM

printf "1. ModusPonens\n"
python analyze_results.py Qwen2_5_3B_modusponens_1hop_ProofsOnly_nodistractor.log
printf "\n\n2. AndIntro\n"
python analyze_results.py Qwen2_5_3B_andintro_1hop_AndIntro_nodistractor.log
printf "\n\n3. AndElim\n"
python analyze_results.py Qwen2_5_3B_andelim_1hop_AndElim_nodistractor.log
printf "\n\n4. OrIntro\n"
python analyze_results.py Qwen2_5_3B_orintro_1hop_OrIntro_nodistractor.log
printf "\n\n5. OrElim\n"
python analyze_results.py Qwen2_5_3B_orelim_1hop_OrElim_nodistractor.log
printf "\n\n6. ProofByContra\n"
python analyze_results.py Qwen2_5_3B_proofbycontra_1hop_ProofByContra_nodistractor.log