#!/bin/bash

echo "Job started at: $(date)"

TOTAL_N=2700
LANG='vi'
TASK="sums"
MODEL_NAME='gpt-4o-mini' # 'gemini/gemini-2.0-flash'

if [[ "$MODEL_NAME" == "gpt-4o-mini" ]]; then
  MODEL_ABB="gpt"
elif [[ "$MODEL_NAME" == "gemini/gemini-2.0-flash" ]]; then
  MODEL_ABB="gemini"
fi

IN_FILE="summaries/ds/${LANG}_sums.jsonl"
OUT_FILE="summaries/ds/mgt/${LANG}_sums_mgt_few1_${MODEL_ABB}.jsonl"
PROMPT_TEMPLATE_FILE="summaries/prompts/few_${LANG}.txt"
FEW_SHOTS_FILE="summaries/prompts/${LANG}/shots_${LANG}.jsonl"
N_SHOTS=1

python mgt/mgt_api.py \
  --total_n $TOTAL_N \
  --lang $LANG \
  --task $TASK \
  --in_file $IN_FILE \
  --out_file $OUT_FILE \
  --model_name $MODEL_NAME \
  --prompt_template_file $PROMPT_TEMPLATE_FILE \
  --few_shots_file $FEW_SHOTS_FILE \
  --n_shots $N_SHOTS 

echo "Job finished at: $(date)"