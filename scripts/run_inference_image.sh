#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sia

cd "$(dirname "$0")/.."

python inference_image_pose.py \
    --image_path extras/dancing.jpg \
    --output_path output_image.jpg \
    --checkpoint_path weights/avak_b16_11.pt \
    --device cuda:0 \
    --conf_thresh 0.5 