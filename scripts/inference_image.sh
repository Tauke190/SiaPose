#!/bin/bash
conda activate sia

cd "$(dirname "$0")/.."

python inference_image.py \
    --image_path extras/man.png \
    --checkpoint_path output/sia_pose_simple1_frames_b16_best.pt\
    --model sia_pose_simple \
    --amp --batch_size 8