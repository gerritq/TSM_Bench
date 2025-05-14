#!/bin/bash
#SBATCH --job-name=crawl-vi
#SBATCH --error=logs/%A-%a.err
#SBATCH --output=logs/%A-%a.out
#SBATCH --time=10:00:00
#SBATCH --array=0-1

# For EN, we use 15 arrays to speed up processing

LANG='vi'

python3 tst/data/2_crawl.py ${LANG}

cat data/${LANG}/temp/*_${LANG}_crawl.jsonl > data/${LANG}/2_${LANG}_crawl.jsonl

