#!/bin/bash

LANG="pt"
IN_FILE="summaries/ds/eval/${LANG}_sums_eval.jsonl"
OUT_FILE="summaries/ds/metrics/${LANG}_sums_eval.jsonl"
PROMPT_TECHNIQUES=("minimal" "instruct" "few1" "few2" "few3") 
TOTAL_N=270

python scorers/brb.py \
    --lang $LANG \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --prompt_techs "${PROMPT_TECHNIQUES[@]}" \
    --total_n $TOTAL_N
