#!/bin/bash
set -e

# echo "Running off-the-shelf detectors ..."
# python generalise/code/ots.py

echo "Running binoculars ..."
python generalise/code/bin_g.py
echo "Run successful."

