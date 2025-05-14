#!/bin/bash

LANG="vi"
SUBSET="extend"
TOTAL_N=270
IN_FILE="paragraphs/ds/eval/${LANG}_paras_${SUBSET}.jsonl"
OUT_FILE="paragraphs/ds/metrics/${LANG}_paras_${SUBSET}.jsonl"
PROMPT_TECHNIQUES=("minimal" "cp" "rag")

python scorers/brb.py \
    --lang $LANG \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --prompt_techs "${PROMPT_TECHNIQUES[@]}" \
    --total_n $TOTAL_N 