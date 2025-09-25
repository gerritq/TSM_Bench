#!/bin/bash
#SBATCH --job-name=fig1
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=nmes_gpu,gpu,interruptible_gpu
#SBATCH --gres=gpu:1
nvidia-smi


LANGS=("en" "pt" "vi")

for LANG in "${LANGS[@]}"; do
    uv run la.py ${LANG}
done
