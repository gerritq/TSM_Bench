#!/bin/bash 

start_time=$(date +%s)
echo "Job started at: $(date)"

nvidia-smi


LANGUAGES=("en" "en_paras" "pt" "vi") 
MODELS=("gpt" "gemini" "qwen" "mistral")
TASK="neutral"
N_SHOTS=5

BATCH_SIZES=(16 32)
LEARNING_RATES=(1e-5 5e-6 1e-6)
EPOCHS=(3 5)

for LANG in "${LANGUAGES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    
      if [ "$LANG" = "en" ]; then
          IN_FILE="tst/ds/mgt/${LANG}_default_mgt_few${N_SHOTS}_${MODEL}.jsonl"
          OUT_FILE="tst/ds/detect/${LANG}_default_mgt_few${N_SHOTS}_${MODEL}_train_hp.jsonl"
      elif [ "$LANG" = "en_paras" ]; then
          IN_FILE="tst/ds/mgt/${LANG}_mgt_few${N_SHOTS}_${MODEL}.jsonl"
          OUT_FILE="tst/ds/detect/${LANG}_mgt_few${N_SHOTS}_${MODEL}_train_hp.jsonl"
      else
          IN_FILE="tst/ds/mgt/${LANG}_mgt_few${N_SHOTS}_${MODEL}.jsonl"
          OUT_FILE="tst/ds/detect/${LANG}_mgt_few${N_SHOTS}_${MODEL}_train_hp.jsonl"
      fi
    
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
            --task "$TASK" \
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

end_time=$(date +%s)
runtime=$((end_time - start_time))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "Job finished at: $(date)"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"