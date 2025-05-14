#!/bin/bash

nvidia-smi

source activate qa_env
python --version

LANG="vi"
SUBSET="extend"
DS="paras"
IN_FILE="paragraphs/ds/eval/${LANG}_paras_${SUBSET}.jsonl"
OUT_FILE="paragraphs/ds/metrics/${LANG}_paras_${SUBSET}.jsonl"

PROMPT_TECHNIQUES=("minimal" "cp" "rag")

python scorers/qafe.py \
    --lang $LANG \
    --ds $DS \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --prompt_techs "${PROMPT_TECHNIQUES[@]}" 
