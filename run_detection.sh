#!/bin/bash
set -e

# Paragraphs
echo "Running detection for Paragraph Writing ...."
echo "  Zero-shot detectors ...."
bash paragraphs/analysis/detect_zero.sh

echo "  Supervised detectors ...."
bash paragraphs/analysis/detect_train_hp.sh

# Summarisation
echo "Running detection for Summarisation ...."
echo "  Zero-shot detectors ...."
bash summaries/analysis/detect_zero.sh

echo "  Supervised detectors ...."
bash summaries/analysis/detect_train_hp.sh

# TST
echo "Running detection for TST ...."
echo "  Zero-shot detectors ...."
bash tst/analysis/detect_zero.sh

echo "  Supervised detectors ...."
bash tst/analysis/detect_train_hp.sh