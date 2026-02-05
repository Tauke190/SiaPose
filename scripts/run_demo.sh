#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sia

cd "$(dirname "$0")/.."

python demo.py -F example1.mp4 -thresh 0.5 -act "walk,run"