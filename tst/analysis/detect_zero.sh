#!/bin/bash
set -e

nvidia-smi

LANGUAGES=("en" "en_paras" "pt" "vi") 
MODELS=("gpt" "gemini" "qwen" "mistral")
TASK="neutral"
N_SHOTS=5
DETECTORS=("binoculars" "llr" "fastdetectgpt_white" "fastdetectgpt_black") # "revise", "gecscore"

for LANG in "${LANGUAGES[@]}"; do
    for MODEL in "${MODELS[@]}"; do

        echo "Processing LANG=${LANG}, TASK=${TASK}, MODEL=${MODEL} at $(date)"


        if [ "$LANG" = "en" ]; then
            IN_FILE="tst/ds/mgt/${LANG}_default_mgt_few${N_SHOTS}_${MODEL}.jsonl"
            OUT_FILE="tst/ds/detect/${LANG}_default_mgt_few${N_SHOTS}_${MODEL}_zero.jsonl"
        elif [ "$LANG" = "en_paras" ]; then
            IN_FILE="tst/ds/mgt/${LANG}_mgt_few${N_SHOTS}_${MODEL}.jsonl"
            OUT_FILE="tst/ds/detect/${LANG}_mgt_few${N_SHOTS}_${MODEL}_zero.jsonl"
        else
            IN_FILE="tst/ds/mgt/${LANG}_mgt_few${N_SHOTS}_${MODEL}.jsonl"
            OUT_FILE="tst/ds/detect/${LANG}/detect/${LANG}_mgt_few${N_SHOTS}_${MODEL}_zero.jsonl"
        fi

        python detectors/detect.py \
          --in_file "${IN_FILE}" \
          --out_file "${OUT_FILE}" \
          --task "${TASK}" \
          --lang "${LANG:0:2}" \
          --detectors "${DETECTORS[@]}"

  done
done
