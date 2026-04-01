#!/bin/bash
# Download the "DL in NLP Spring 2019 Classification" dataset from Kaggle.
# Requires the Kaggle API: pip install kaggle
# and a valid ~/.kaggle/kaggle.json credentials file.
#
# Usage: bash data/download_data.sh

set -e

echo "Downloading dataset from Kaggle..."
kaggle datasets download -d datasnaek/clickbait -p data/ --unzip

echo "Done. Files saved in data/"
ls data/
