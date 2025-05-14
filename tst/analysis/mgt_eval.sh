#!/bin/bash

# Select paras/default for EN here (first if condition)

mkdir -p tst/ds/eval_paras

mkdir -p tst/ds/eval


LANG="vi" # en_default, pt_default, vi_mpairs
if [ "$LANG" = "en" ]; then
   #SUBSET="default"
   SUBSET="paras"
elif [ "$LANG" = "pt" ]; then
   SUBSET="default"
elif [ "$LANG" = "vi" ]; then
   SUBSET="mpairs"
fi

DS="tst"
TOTAL_N=270
IN_FILE="tst/ds/eval/${LANG}_${SUBSET}_eval.jsonl"

if [ "$LANG" = "en" ] && [ "$SUBSET" = "paras" ]; then
   OUT_FILE="tst/ds/eval_paras/${LANG}_eval.jsonl"
else
   OUT_FILE="tst/ds/eval/${LANG}_eval.jsonl"
fi

PROMPT_DIR="tst/prompts/${LANG:0:2}"
PROMPT_TECHNIQUES=("minimal" "instruct" "few1" "few2" "few3" "few4" "few5")

if [ "$LANG" = "en" ]; then
   FS_FILE="tst/prompts/${LANG:0:2}/shots_${SUBSET}_${LANG:0:2}.jsonl"
else
   FS_FILE="tst/prompts/${LANG:0:2}/shots_${LANG:0:2}.jsonl"
fi

python mgt/mgt.py \
    --lang $LANG \
    --ds $DS \
    --subset $SUBSET \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --prompt_dir $PROMPT_DIR \
    --few_shots_file $FS_FILE \
    --prompt_techs "${PROMPT_TECHNIQUES[@]}"  \
    --total_n $TOTAL_N



