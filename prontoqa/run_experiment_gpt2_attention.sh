#!/bin/bash

#SBATCH --partition=general-gpu               # Name of Partition
#SBATCH --gres=gpu:1                          # Request 1 GPU card for the job
#SBATCH --constraint=epyc64
#SBATCH --constraint=a100
#SBATCH --output=R-%x_%j.out

module purge

source /gpfs/homefs1/ych22001/miniconda3/etc/profile.d/conda.sh

conda activate HF_LLM

python run_experiment.py --model-name gpt2_attention --distractors none --test-distractors none --num-trials 1 --few-shot-examples 3 --proofs-only --max-hops 1 --deduction-rule ModusPonens

python run_experiment.py --model-name gpt2_attention --distractors none --test-distractors none --num-trials 1 --few-shot-examples 3 --proofs-only --max-hops 1 --deduction-rule AndIntro

python run_experiment.py --model-name gpt2_attention --distractors none --test-distractors none --num-trials 1 --few-shot-examples 3 --proofs-only --max-hops 1 --deduction-rule AndElim

python run_experiment.py --model-name gpt2_attention --distractors none --test-distractors none --num-trials 1 --few-shot-examples 3 --proofs-only --max-hops 1 --deduction-rule OrIntro

python run_experiment.py --model-name gpt2_attention --distractors none --test-distractors none --num-trials 1 --few-shot-examples 3 --proofs-only --max-hops 1 --deduction-rule OrElim

python run_experiment.py --model-name gpt2_attention --distractors none --test-distractors none --num-trials 1 --few-shot-examples 3 --proofs-only --max-hops 1 --deduction-rule ProofByContra