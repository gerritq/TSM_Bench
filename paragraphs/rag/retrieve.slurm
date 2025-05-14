#!/bin/bash
start_time=$(date +%s)
echo "Job started at: $(date)"

nvidia-smi

LANGS=("pt")
SUBSETS=("extend")

python retrieve2.py --subsets "${SUBSETS[@]}" --langs "${LANGS[@]}"