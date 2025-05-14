#!/bin/bash
set -e

mkdir -p summaries/ds/metrics
mkdir -p paragraphs/ds/metrics
mkdir -p tst/ds/metrics

# echo "Running prompt eval for Summarisation ..."
# echo "  Obtaining n-gram and bertscore metrics ..."
# bash summaries/metrics/brb.sh
# echo "  Obtaining QAFacteval scores ..."
# bash summaries/metrics/qafe.sh

# echo "Running prompt eval for Paragraph Writing (continutation)..."
# echo "  Obtaining n-gram and bertscore metrics ..."
# bash paragraphs/metrics/brb.sh
# echo "  Obtaining QAFacteval scores ..."
# bash paragraphs/metrics/qafe.sh

echo "Running prompt eval for TST ..."
bash tst/metrics/eval.sh