#!/bin/bash
conda activate sia

cd "$(dirname "$0")/.."

python inference_image.py \
    --image_path extras/tadasana.jpg \
    --checkpoint_path output/sia_pose_simple_b16_best.pt \
    --model sia_pose_simple \
