#!/bin/bash

#SBATCH --partition=general               # Name of Partition
#SBATCH --constraint=epyc64
#SBATCH --output=R-%x_%j.out

module purge

source /gpfs/homefs1/ych22001/miniconda3/etc/profile.d/conda.sh

conda activate HF_LLM

python run_experiment.py --model-name dummy --distractors none --test-distractors none --num-trials 100 --few-shot-examples 3 --proofs-only --max-hops 1 --deduction-rule ModusPonens

python run_experiment.py --model-name dummy --distractors none --test-distractors none --num-trials 100 --few-shot-examples 3 --proofs-only --max-hops 1 --deduction-rule AndIntro

python run_experiment.py --model-name dummy --distractors none --test-distractors none --num-trials 100 --few-shot-examples 3 --proofs-only --max-hops 1 --deduction-rule AndElim

python run_experiment.py --model-name dummy --distractors none --test-distractors none --num-trials 100 --few-shot-examples 3 --proofs-only --max-hops 1 --deduction-rule OrIntro

python run_experiment.py --model-name dummy --distractors none --test-distractors none --num-trials 100 --few-shot-examples 3 --proofs-only --max-hops 1 --deduction-rule OrElim

python run_experiment.py --model-name dummy --distractors none --test-distractors none --num-trials 100 --few-shot-examples 3 --proofs-only --max-hops 1 --deduction-rule ProofByContra