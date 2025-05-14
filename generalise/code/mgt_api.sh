#!/bin/bash

echo "Job started at: $(date)"


PROMPT_DIR="generalise/code/prompts"
DATA_DIR="generalise/data/ds/external"

TOTAL_N=2700
LANGS=("en" "pt" "vi")
TASK="external"
MODELS=("gpt-4o-mini")

if [[ "$LANG" == "en" ]]; then
  DATASETS=("wiki" "cnndm" "yelp")
else
  DATASETS=("wiki" "news" "reviews")
fi

for LANG in "${LANGS[@]}"; do
  for SET in "${DATASETS[@]}"; do
    for MODEL_NAME in "${MODELS[@]}"; do

    IN_FILE="${DATA_DIR}/${SET}_${LANG}.jsonl"
    OUT_FILE="${DATA_DIR}/mgt/${SET}_${LANG}_${MODEL_ABB}.jsonl"
    PROMPT_TEMPLATE_FILE="${PROMPT_DIR}/${SET}_${LANG}.txt"

    python mgt/mgt_api.py \
      --total_n $TOTAL_N \
      --lang $LANG \
      --task $TASK \
      --in_file $IN_FILE \
      --out_file $OUT_FILE \
      --model_name $MODEL_NAME \
      --prompt_template_file $PROMPT_TEMPLATE_FILE

    done
  done
done

echo "Job finished at: $(date)"

end_time=$(date +%s)
runtime=$((end_time - start_time))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "Job finished at: $(date)"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"