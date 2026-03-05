#!/bin/bash
conda activate sia

cd "$(dirname "$0")/.."

python inference_image.py \
    --image_path extras/four.jpg \
    --checkpoint_path weights/sia_ROIAlign_2.pt\
    --model sia_pose_coco 