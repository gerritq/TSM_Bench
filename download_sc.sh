#!/bin/bash

cd tst/ds

FILE_ID="1veyTdjbJaCp_nFLelkkxG1zoDtRxiYG7"
ZIP_NAME="sc_models.zip"

if [ ! -d "sc" ]; then
  echo "Downloading model archive from Google Drive..."
  gdown --id "$FILE_ID" -O "$ZIP_NAME"
  echo "Unzipping..."
  unzip -q "$ZIP_NAME"
  rm "$ZIP_NAME"
  echo "Models downloaded and extracted to tst/ds/sc"
else
  echo "Model directory already exists. Skipping download."
fi