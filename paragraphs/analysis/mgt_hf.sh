#!/bin/bash

start_time=$(date +%s)
echo "Job started at: $(date)"

nvidia-smi

TOTAL_N=2700
BATCH_SIZE=4
CODE_DIR="paragraphs"
DATA_DIR="paragraphs/ds"

LANGS=("en" "pt" "vi")
TASKS=("extend") # first
MODELS=("mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen2.5-7B-Instruct")

for LANG in "${LANGS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for MODEL_NAME in "${MODELS[@]}"; do

      if [[ "$MODEL_NAME" == "Qwen/Qwen2.5-7B-Instruct" ]]; then
        MODEL_ABB="qwen"
      elif [[ "$MODEL_NAME" == "mistralai/Mistral-7B-Instruct-v0.3" ]]; then
        MODEL_ABB="mistral"
      fi

      echo "Running combination: LANG=$LANG, TASK=$TASK, MODEL=$MODEL_ABB"

      nvidia-smi

      IN_FILE="${DATA_DIR}/${LANG}_paras_context_${TASK}.jsonl" # we will provide this shortly
      OUT_FILE="${DATA_DIR}/mgt/${LANG}_paras_rag_${TASK}_${MODEL_ABB}.jsonl"
      PROMPT_TEMPLATE_FILE="${CODE_DIR}/prompts/${LANG}/${TASK}_rag_${LANG}.txt"

      python mgt/mgt_hf.py \
        --total_n "$TOTAL_N" \
        --lang "$LANG" \
        --task "$TASK" \
        --in_file "$IN_FILE" \
        --out_file "$OUT_FILE" \
        --model_name "$MODEL_NAME" \
        --prompt_template_file "$PROMPT_TEMPLATE_FILE" \
        --batch_size "$BATCH_SIZE"

      echo "Finished LANG=$LANG, TASK=$TASK, MODEL=$MODEL_ABB at: $(date)"
      echo "--------------------------------------------"
      torch.cuda.empty_cache
    done
  done
done

end_time=$(date +%s)
runtime=$((end_time - start_time))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "Job finished at: $(date)"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"
