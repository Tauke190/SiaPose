#!/bin/bash
# Visualization script for COCO validation predictions vs ground truth

conda activate sia

# cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=2 python visualize_coco.py \
    --checkpoint_path weights/sia_pose_coco_roi_best_b16_best.pt \
    --model sia_pose_coco_best \
    --size b16 \
    --det_tokens 20 \
    --pose_layers 2 \
    --num_frames 1 \
    --num_images 20 \
    --data_root /mnt/SSD2/coco2017/val2017 \
    --ann_file /mnt/SSD2/coco2017/annotations/person_keypoints_val2017.json \
    --output_dir vis_results/coco_val \
    --conf_thresh 0.5 \
    --kp_conf_thresh 0.3 \
    --img_height 480 \
    --img_width 620
