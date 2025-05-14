#!/bin/bash

start_time=$(date +%s)
echo "Job started at: $(date)"

nvidia-smi

LANGUAGES=("en" "pt" "vi") 
MODELS=("gpt" "qwen")
TASK="train"
DATA_PREFIX=("first_sums")

BATCH_SIZES=(16 32)
LEARNING_RATES=(1e-5 5e-6 1e-6)
EPOCHS=(3 5)

DATA_DIR="generalise/data"

for LANG in "${LANGUAGES[@]}"; do

  if [[ "$LANG" == "pt" || "$LANG" == "vi" ]]; then
    DATA_PREFIX=("first_sums" "wiki" "news" "reviews")
  else
    DATA_PREFIX=("first_sums" "wiki" "cnndm" "yelp")
  fi

  for DP in "${DATA_PREFIX[@]}"; do
    for MODEL in "${MODELS[@]}"; do


    if [[ "$DP" == "arxiv" || "$DP" == "wiki" || "$DP" == "cnndm" || "$DP" == "yelp" || "$DP" == "reviews" || "$DP" == "news" ]]; then
        IN_FILE="${DATA_DIR}/ds/external/mgt/${DP}_${LANG}_${MODEL}.jsonl"
    else
        IN_FILE="${DATA_DIR}/ds/our/${DP}_${LANG}_${MODEL}.jsonl" # for our data only
    fi
      
      OUT_FILE="${DATA_DIR}/hp/${DP}_${MODEL}_${LANG}.jsonl"
      LEADER_FILE="${DATA_DIR}/hp/leader_${DP}_${MODEL}_${LANG}.jsonl"
      BEST_MODEL_DIR="${DATA_DIR}/hp/best_${DP}_${MODEL}_${LANG}"
      
    
      for BS in "${BATCH_SIZES[@]}"; do
        for LR in "${LEARNING_RATES[@]}"; do
          for EP in "${EPOCHS[@]}"; do
      
            echo " "
            echo " "
            echo "Processing LANG=${LANG}, DP=${DP}, MODEL=${MODEL}, TASK=${TASK}, BS=${BS}, LR=${LR}, EP=${EP}"
            
            run_start_time=$(date +%s)

            python detectors/train_hp_g.py \
              --in_file "${IN_FILE}" \
              --out_file "${OUT_FILE}" \
              --lead_file "${LEADER_FILE}" \
              --best_model_dir "${BEST_MODEL_DIR}" \
              --models "microsoft/mdeberta-v3-base"  \
              --task $TASK \
              --batch_size $BS \
              --learning_rate $LR \
              --epochs $EP \
              --date $start_time \
              --save

            run_end_time=$(date +%s)
            run_time=$((run_end_time - run_start_time))
            
            run_hours=$((run_time / 3600))
            run_minutes=$(((run_time % 3600) / 60))
            run_seconds=$((run_time % 60))

            echo "Completed LANG=${LANG}, TASK=${TASK}, BS=${BS}, LR=${LR}, EP=${EP}"
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