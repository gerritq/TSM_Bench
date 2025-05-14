#!/bin/bash

start_time=$(date +%s)
echo "Job started at: $(date)"

nvidia-smi

DATA_DIR="paragraphs/ds"
LANGS=("en" "pt" "vi")
TASKS=("first" "extend")
MODELS=("gpt" "gemini" "qwen" "mistral")
DETECTORS=("binoculars" "llr" "fastdetectgpt_white" "fastdetectgpt_black") # "revise", "gecscore"

for LANG in "${LANGS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

      echo " "
      echo "Running LANG=${LANG}, TASK=${TASK}, MODEL=${MODEL}"

      iter_start=$(date +%s)

      IN_FILE="${DATA_DIR}/mgt/${LANG}_paras_rag_${TASK}_${MODEL}.jsonl"
      OUT_FILE="${DATA_DIR}/detect/${LANG}_paras_rag_${TASK}_${MODEL}_zero.jsonl"

      python detectors/detect.py \
        --in_file "${IN_FILE}" \
        --out_file "${OUT_FILE}" \
        --task "${TASK}" \
        --lang "${LANG}" \
        --detectors "${DETECTORS[@]}"

      iter_end=$(date +%s)
      iter_runtime=$((iter_end - iter_start))

      iter_h=$((iter_runtime / 3600))
      iter_m=$(((iter_runtime % 3600) / 60))
      iter_s=$((iter_runtime % 60))

      echo "Iteration LANG=${LANG}, TASK=${TASK}, MODEL=${MODEL} finished in ${iter_h}h ${iter_m}m ${iter_s}s"
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
