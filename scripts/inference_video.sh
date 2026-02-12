#!/bin/bash
conda activate sia

cd "$(dirname "$0")/.."

python inference_video.py \
    --video_path extras/figure_skating.mp4\
    --checkpoint_path output/sia_pose_simple1_frames_b16_best.pt \
    --model sia_pose_simple \
    --conf_thresh 0.5 \
    --kp_conf_thresh 0.3 \
    --amp
