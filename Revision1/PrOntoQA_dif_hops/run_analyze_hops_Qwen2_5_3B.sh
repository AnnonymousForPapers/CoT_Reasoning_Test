#!/bin/bash

#SBATCH --partition=general               # Name of Partition
#SBATCH --output=R-%x_%j.out

module purge

source /gpfs/homefs1/ych22001/miniconda3/etc/profile.d/conda.sh

conda activate HF_LLM

printf "1. ModusPonens\n"
printf "\nHop 1\n"
python analyze_results.py Qwen2_5_3B_modusponens_1hop_ProofsOnly_3shot_nodistractor.log
printf "\nHop 2\n"
python analyze_results.py Qwen2_5_3B_modusponens_2hop_ProofsOnly_3shot_nodistractor.log
printf "\nHop 3\n"
python analyze_results.py Qwen2_5_3B_modusponens_3hop_ProofsOnly_3shot_nodistractor.log
printf "\nHop 4\n"
python analyze_results.py Qwen2_5_3B_modusponens_4hop_ProofsOnly_3shot_nodistractor.log
printf "\nHop 5\n"
python analyze_results.py Qwen2_5_3B_modusponens_5hop_ProofsOnly_3shot_nodistractor.log

printf "\n\n2. AndIntro\n"
printf "\nHop 1\n"
python analyze_results.py Qwen2_5_3B_andintro_1hop_AndIntro_3shot_nodistractor.log
printf "\nHop 2\n"
python analyze_results.py Qwen2_5_3B_andintro_2hop_AndIntro_3shot_nodistractor.log
printf "\nHop 3\n"
python analyze_results.py Qwen2_5_3B_andintro_3hop_AndIntro_3shot_nodistractor.log
printf "\nHop 4\n"
python analyze_results.py Qwen2_5_3B_andintro_4hop_AndIntro_3shot_nodistractor.log
printf "\nHop 5\n"
python analyze_results.py Qwen2_5_3B_andintro_5hop_AndIntro_3shot_nodistractor.log

printf "\n\n3. AndElim\n"
printf "\nHop 1\n"
python analyze_results.py Qwen2_5_3B_andelim_1hop_AndElim_3shot_nodistractor.log
printf "\nHop 2\n"
python analyze_results.py Qwen2_5_3B_andelim_2hop_AndElim_3shot_nodistractor.log
printf "\nHop 3\n"
python analyze_results.py Qwen2_5_3B_andelim_3hop_AndElim_3shot_nodistractor.log
printf "\nHop 4\n"
python analyze_results.py Qwen2_5_3B_andelim_4hop_AndElim_3shot_nodistractor.log
printf "\nHop 5\n"
python analyze_results.py Qwen2_5_3B_andelim_5hop_AndElim_3shot_nodistractor.log

printf "\n\n4. OrIntro\n"
printf "\nHop 1\n"
python analyze_results.py Qwen2_5_3B_orintro_1hop_OrIntro_3shot_nodistractor.log
printf "\nHop 2\n"
python analyze_results.py Qwen2_5_3B_orintro_2hop_OrIntro_3shot_nodistractor.log
printf "\nHop 3\n"
python analyze_results.py Qwen2_5_3B_orintro_3hop_OrIntro_3shot_nodistractor.log
printf "\nHop 4\n"
python analyze_results.py Qwen2_5_3B_orintro_4hop_OrIntro_3shot_nodistractor.log
printf "\nHop 5\n"
python analyze_results.py Qwen2_5_3B_orintro_5hop_OrIntro_3shot_nodistractor.log