#!/bin/bash

model1_dir="generalise/data/hp/best_first_sums_gpt_en"
model2_dir="generalise/data/hp/best_wiki_gpt_en"

for dir in "$model1_dir" "$model2_dir"; do
  if [ ! -d "$dir" ]; then
    echo "Model directory does not exist: $dir"
    echo "You likely did not download the models. Please run download_models.sh"
    exit 1
  fi
done

echo "Running shap values analysis ..."
python generalise/code/shap_vals.py
echo "Plot created to the assets folders"