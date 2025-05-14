#!/bin/bash

echo "Job started at: $(date)"

TASK="tst"
TOTAL_N=2700
LANG='en'
SET="paras"
N_SHOTS=5
MODEL_NAME='gemini/gemini-2.0-flash' #'gpt-4o-mini'

IN_FILE="tst/ds/4_${LANG}_${SET}.jsonl"

if [[ "$MODEL_NAME" == "gpt-4o-mini" ]]; then
  MODEL_ABB="gpt"
elif [[ "$MODEL_NAME" == "gemini/gemini-2.0-flash" ]]; then
  MODEL_ABB="gemini"
fi

if [ "$LANG" = "en" ]; then
   OUT_FILE="tst/ds/mgt/${LANG}_${SET}_mgt_few${N_SHOTS}_${MODEL_ABB}.jsonl"
   PROMPT_TEMPLATE_FILE="tst/prompts/${LANG}/few_${SET}_${LANG}.txt"
   FS_FILE="tst/prompts/${LANG}/shots_${SET}_${LANG}.jsonl"
else
   OUT_FILE="tst/ds/mgt/${LANG}_mgt_few${N_SHOTS}_${MODEL_ABB}.jsonl"
   PROMPT_TEMPLATE_FILE="tst/prompts/${LANG}/few_${LANG}.txt"
   FS_FILE="tst/prompts/${LANG}/shots_${LANG}.jsonl"
fi

python mgt/mgt_api.py \
  --total_n $TOTAL_N \
  --lang $LANG \
  --task $TASK \
  --in_file $IN_FILE \
  --out_file $OUT_FILE \
  --model_name $MODEL_NAME \
  --prompt_template_file $PROMPT_TEMPLATE_FILE \
  --few_shots_file $FS_FILE \
  --n_shots $N_SHOTS 

echo "Job finished at: $(date)"