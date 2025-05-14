#!/bin/bash 

start_time=$(date +%s)
echo "Job started at: $(date)"

nvidia-smi

LANGUAGES=("en" "pt" "vi")
MODELS=("gpt" "gemini" "qwen" "mistral")
TASKS=("first" "extend")

BATCH_SIZES=(16 32)
LEARNING_RATES=(1e-5 5e-6 1e-6)
EPOCHS=(3 5)

DATA_DIR="paragraphs/ds"

for LANG in "${LANGUAGES[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      
      IN_FILE="${DATA_DIR}/mgt/${LANG}_paras_rag_${TASK}_${MODEL}.jsonl"
      OUT_FILE="${DATA_DIR}/detect/${LANG}_paras_rag_${TASK}_${MODEL}_train_hp.jsonl"
      
      for BS in "${BATCH_SIZES[@]}"; do
        for LR in "${LEARNING_RATES[@]}"; do
          for EP in "${EPOCHS[@]}"; do
      
            echo " "
            echo "Processing LANG=${LANG}, TASK=${TASK}, MODEL=${MODEL}, BS=${BS}, LR=${LR}, EP=${EP}"
              
            run_start_time=$(date +%s)

            python detectors/train_hp.py \
              --in_file "${IN_FILE}" \
              --out_file "${OUT_FILE}" \
              --models "microsoft/mdeberta-v3-base" "FacebookAI/xlm-roberta-base" \
              --task $TASK \
              --batch_size $BS \
              --learning_rate $LR \
              --epochs $EP \
              --date $start_time

            run_end_time=$(date +%s)
            run_time=$((run_end_time - run_start_time))
            
            run_hours=$((run_time / 3600))
            run_minutes=$(((run_time % 3600) / 60))
            run_seconds=$((run_time % 60))

            echo "Completed LANG=${LANG}, TASK=${TASK}, MODEL=${MODEL}, BS=${BS}, LR=${LR}, EP=${EP}"
            echo "Run time: ${run_hours}h ${run_minutes}m ${run_seconds}s"
            echo "---------------------------------------------------"
          done
        done
      done
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