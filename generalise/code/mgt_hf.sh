#!/bin/bash

start_time=$(date +%s)
echo "Job started at: $(date)"

nvidia-smi

TOTAL_N=2700
BATCH_SIZE=16
PROMPT_DIR="generalise/code/prompts"
DATA_DIR="generalise/data/ds/external"

LANGS=("en" "pt" "vi")
TASK="external"
MODELS=("Qwen/Qwen2.5-7B-Instruct")
DATASETS=("reviews")
MODEL_ABB="qwen"

for LANG in "${LANGS[@]}"; do
  for SET in "${DATASETS[@]}"; do
    for MODEL_NAME in "${MODELS[@]}"; do

      echo "Running combination: LANG=$LANG, SET=$SET, MODEL=$MODEL_ABB"

      IN_FILE="${DATA_DIR}/${SET}_${LANG}.jsonl"
      OUT_FILE="${DATA_DIR}/mgt/${SET}_${LANG}_${MODEL_ABB}.jsonl"
      PROMPT_TEMPLATE_FILE="${PROMPT_DIR}/${SET}_${LANG}.txt"

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
