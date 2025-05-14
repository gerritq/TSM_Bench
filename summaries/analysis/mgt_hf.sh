#!/bin/bash

echo "Job started at: $(date)"

nvidia-smi

mkdir -p summaries/ds/mgt

TOTAL_N=2700
LANG='vi'
TASK="sums"
IN_FILE="summaries/ds/${LANG}_sums.jsonl"
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" # 'Qwen/Qwen2.5-7B-Instruct'

if [[ "$MODEL_NAME" == "Qwen/Qwen2.5-7B-Instruct" ]]; then
  MODEL_ABB="qwen"
elif [[ "$MODEL_NAME" == "mistralai/Mistral-7B-Instruct-v0.3" ]]; then
  MODEL_ABB="mistral"
fi

N_SHOTS=1
OUT_FILE="summaries/ds/mgt/${LANG}_sums_mgt_few${N_SHOTS}_${MODEL_ABB}.jsonl"
PROMPT_TEMPLATE_FILE="summaries/prompts/${LANG}/few_${LANG}.txt"
FEW_SHOTS_FILE="summaries/prompts/${LANG}/shots_${LANG}.jsonl"
BATCH_SIZE=1

python mgt/mgt_hf.py \
  --total_n $TOTAL_N \
  --lang $LANG \
  --task $TASK \
  --in_file $IN_FILE \
  --out_file $OUT_FILE \
  --model_name $MODEL_NAME \
  --prompt_template_file $PROMPT_TEMPLATE_FILE \
  --few_shots_file $FEW_SHOTS_FILE \
  --n_shots $N_SHOTS \
  --batch_size $BATCH_SIZE

echo "Job finished at: $(date)"