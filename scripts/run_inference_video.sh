#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sia

cd "$(dirname "$0")/.."

python inference_video.py \
    --video_path videoplayback.mp4 \
    --output_path output_video.mp4 \
    --checkpoint_path weights/avak_b16_11.pt \
    --device cuda:0 \
    --actions standing shakinghands \
    --thresh 0.5 \
    --show_all_scores 