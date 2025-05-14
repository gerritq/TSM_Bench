#!/bin/bash

# change lang here
LANG="vi"
IN_FILE="data/${LANG}/1_${LANG}_latest_articles.jsonl"
OUT_FILE="data/${LANG}/2_${LANG}_html.jsonl"
TOTAL_COUNT=100000

python collection/2_query2.py --lang $LANG \
    --in_file $IN_FILE \
    --out_file $OUT_FILE \
    --total_count $TOTAL_COUNT

