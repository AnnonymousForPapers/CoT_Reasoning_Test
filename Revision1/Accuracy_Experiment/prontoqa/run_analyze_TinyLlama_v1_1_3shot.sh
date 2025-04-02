#!/bin/bash

#SBATCH --partition=general               # Name of Partition
#SBATCH --output=R-%x_%j.out

module purge

source /gpfs/homefs1/ych22001/miniconda3/etc/profile.d/conda.sh

conda activate HF_LLM

printf "1. ModusPonens\n"
python analyze_results.py TinyLlama_v1_1_modusponens_1hop_ProofsOnly_3shot_nodistractor.log
printf "\n\n2. AndIntro\n"
python analyze_results.py TinyLlama_v1_1_andintro_1hop_AndIntro_3shot_nodistractor.log
printf "\n\n3. AndElim\n"
python analyze_results.py TinyLlama_v1_1_andelim_1hop_AndElim_3shot_nodistractor.log
printf "\n\n4. OrIntro\n"
python analyze_results.py TinyLlama_v1_1_orintro_1hop_OrIntro_3shot_nodistractor.log
printf "\n\n5. OrElim\n"
python analyze_results.py TinyLlama_v1_1_orelim_1hop_OrElim_3shot_nodistractor.log
printf "\n\n6. ProofByContra\n"
python analyze_results.py TinyLlama_v1_1_proofbycontra_1hop_ProofByContra_3shot_nodistractor.log
