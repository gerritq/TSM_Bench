#!/bin/bash


echo "Job started at: $(date)"

TOTAL_N=2700
LANG='en' # select lang here
TASK="extend" # select task here between first and extend
MODEL_NAME='gpt-4o-mini' # 'gemini/gemini-2.0-flash'

CODE_DIR="paragraphs"
DATA_DIR="paragraphs/ds"
IN_FILE="${DATA_DIR}/${LANG}/ds/${LANG}_paras_context_${TASK}.jsonl"
PROMPT_TEMPLATE_FILE="${CODE_DIR}/prompts/${LANG}/${TASK}_rag_${LANG}.txt"

if [[ "$MODEL_NAME" == "gpt-4o-mini" ]]; then
  MODEL_ABB="gpt"
elif [[ "$MODEL_NAME" == "gemini/gemini-2.0-flash" ]]; then
  MODEL_ABB="gemini"
fi

OUT_FILE="${DATA_DIR}/mgt/${LANG}_paras_rag_${TASK}_${MODEL_ABB}.jsonl"

python mgt/mgt_api.py \
  --total_n $TOTAL_N \
  --lang $LANG \
  --task $TASK \
  --in_file $IN_FILE \
  --out_file $OUT_FILE \
  --model_name $MODEL_NAME \
  --prompt_template_file $PROMPT_TEMPLATE_FILE

echo "Job finished at: $(date)"