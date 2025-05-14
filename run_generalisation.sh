#!/bin/bash

model1_dir="generalise/data/hp_len/best_first_sums_gpt_en"

for dir in "$model1_dir"; do
  if [ ! -d "$dir" ]; then
    echo "Model directory does not exist: $dir"
    echo "You likely did not download the models. Please run download_models.sh"
    exit 1
  fi
done

echo "Running the full analysis ..."
echo "If you are interested in specific configurations, please set them in file 'generalise/code/generalise.sh'"
bash generalise/code/generalise.sh
echo "Plot created to the assets folders"