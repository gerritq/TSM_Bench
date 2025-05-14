#!/bin/bash


LANGUAGES=("en" "pt" "vi")

for LANG in "${LANGUAGES[@]}"; do
    python gen_sums.py "$LANG"
done