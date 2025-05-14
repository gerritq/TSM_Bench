#!/bin/bash

LANG="vi"
DS="sums"
N=270

mkdir -p summaries/ds/eval


IN_FILE="summaries/ds/${LANG}_sums_eval.jsonl"
OUT_FILE="summaries/ds/eval/${LANG}_sums_eval.jsonl"
PROMPT_DIR="summaries/prompts/${LANG}"
FS_FILE="summaries/prompts/${LANG}/shots_${LANG}.jsonl"
PROMPT_TECHNIQUES=("minimal" "instruct" "few1" "few2" "few3")

python mgt/mgt.py \
    --lang $LANG \
    --ds $DS \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --prompt_dir $PROMPT_DIR \
    --few_shots_file $FS_FILE \
    --prompt_techs "${PROMPT_TECHNIQUES[@]}"  \
    --n $N
