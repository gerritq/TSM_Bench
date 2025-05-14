#!/bin/bash

LANG="vi"

python3 tst/data/3_process.py $LANG

# If splitting to arrays, run this 
# cat data/${LANG}/temp/j*_${LANG}_proc.jsonl > data/${LANG}/temp/3_${LANG}_proc.jsonl
