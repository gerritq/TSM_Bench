#!/bin/bash 

start_time=$(date +%s)
echo "Job started at: $(date)"

nvidia-smi

LANGUAGES=("en" "pt" "vi") 
OUT_DIR="generalise/data/detect"
DATA_DIR="generalise/data"
TASK="generalise"
N=900

for LANG in "${LANGUAGES[@]}"; do

    python generalise/code/generalise.py \
      --out_dir "${OUT_DIR}" \
      --lang "${LANG}" \
      --eval_n ${N}
done

end_time=$(date +%s)
runtime=$((end_time - start_time))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "Job finished at: $(date)"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"