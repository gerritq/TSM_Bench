#!/bin/bash


nvidia-smi

LANGUAGE="vi"
MODELS=("FacebookAI/xlm-roberta-base" "google-bert/bert-base-multilingual-cased") # "FacebookAI/xlm-roberta-base" 
DATASUBSETS=("mpairs")
LEARNING_RATES=(5e-5 1e-5 5e-6 1e-6)
BATCH_SIZES=(16 32)
EPOCHS=(2 4 6)
WEIGHT_DECAY=(0 0.01)
SEEDS=(42 2025)
WARMUP_RATIO=.1

start_time=$(date +%s)

for MODEL in "${MODELS[@]}"; do
  for DATASUBSET in "${DATASUBSETS[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for EPOCH in "${EPOCHS[@]}"; do
          for WD in "${WEIGHT_DECAY[@]}"; do
            for SEED in "${SEEDS[@]}"; do

              run_start_time=$(date +%s)
              python3 test/sc/train.py \
                --language $LANGUAGE \
                --dsubset $DATASUBSET \
                --model $MODEL \
                --epochs $EPOCH \
                --batch_size $BATCH_SIZE \
                --learning_rate $LR \
                --weight_decay $WD \
                --warmup_ratio $WARMUP_RATIO \
                --seed $SEED  2>&1 
              run_end_time=$(date +%s)
              elapsed_time=$((run_end_time - run_start_time))

              if [[ $? -ne 0 ]]; then
                echo "Erros in training ..."
                exit 1
              fi

              echo "Experiment completed: model=$MODEL, dataset=$DATASUBSET, lr=$LR, batch_size=$BATCH_SIZE, epochs=$EPOCH, weight_decay=$WD, seed=$SEED"
              echo "Run time: $(($elapsed_time / 60)) minutes and $(($elapsed_time % 60)) seconds"
              #exit 0
            done
          done
        done
      done
    done
  done
done

end_time=$(date +%s)
total_elapsed_time=$((end_time - start_time))
echo "Total runtime: $(($total_elapsed_time / 60)) minutes and $(($total_elapsed_time % 60)) seconds"
