#!/bin/bash

LANG="vi" # 
DS="paras" # keep this
SUBSET="extend"
N=270

mkdir -p paragraphs/ds/eval/


IN_FILE="paragraphs/ds/${LANG}_paras_context_${SUBSET}.jsonl" # we will upload this data soon
OUT_FILE="paragraphs/ds/eval/${LANG}_paras_${SUBSET}.jsonl"
PROMPT_DIR="paragraphs/prompts/${LANG}"
PROMPT_TECHNIQUES=("minimal" "cp" "rag")

python mgt/mgt.py \
    --lang $LANG \
    --ds $DS \
    --subset $SUBSET \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --prompt_dir $PROMPT_DIR \
    --prompt_techs "${PROMPT_TECHNIQUES[@]}" \
    --n $N