#!/bin/bash

echo "Job started at: $(date)"

nvidia-smi


TASK="tst"
TOTAL_N=2700
LANG='vi'
SET="mpairs"
N_SHOTS=5
BATCH_SIZE=8

MODEL_NAME='Qwen/Qwen2.5-7B-Instruct' # "mistralai/Mistral-7B-Instruct-v0.3"


IN_FILE="tst/ds/4_${LANG}_${SET}.jsonl"

if [[ "$MODEL_NAME" == "Qwen/Qwen2.5-7B-Instruct" ]]; then
  MODEL_ABB="qwen"
elif [[ "$MODEL_NAME" == "mistralai/Mistral-7B-Instruct-v0.3" ]]; then
  MODEL_ABB="mistral"
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


python mgt_hf.py \
  --total_n $TOTAL_N \
  --lang $LANG \
  --task $TASK \
  --in_file $IN_FILE \
  --out_file $OUT_FILE \
  --model_name $MODEL_NAME \
  --prompt_template_file $PROMPT_TEMPLATE_FILE \
  --few_shots_file $FS_FILE \
  --n_shots $N_SHOTS \
  --batch_size $BATCH_SIZE

echo "Job finished at: $(date)"