#!/bin/bash

cd generalise/ds

FILE_ID="1ZKNkMJjhNEaAfYVeyw14gc7zwgzzCiZz"
ZIP_NAME="hp_len.zip"

if [ ! -d "hp_len" ]; then
  echo "Downloading model archive from Google Drive..."
  gdown --id "$FILE_ID" -O "$ZIP_NAME"
  echo "Unzipping..."
  unzip -q "$ZIP_NAME"
  rm "$ZIP_NAME"
  echo "Models downloaded and extracted to generalise/ds/hp_len"
else
  echo "Model directory already exists. Skipping download."
fi
