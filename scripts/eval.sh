#!/bin/bash
# Evaluate SIA Pose model on COCO val2017
#
# Usage:
#   bash scripts/eval.sh
#
# Override defaults with environment variables:
#   CHECKPOINT=weights/my_model.pt MODEL=sia_pose bash scripts/eval.sh

CHECKPOINT=${CHECKPOINT:-"output/sia_pose_simple_b16_best.pt"}
MODEL=${MODEL:-"sia_pose_simple"}
SIZE=${SIZE:-"b16"}
COCO_ROOT=${COCO_ROOT:-"/mnt/SSD2/coco2017/images"}
BATCH_SIZE=${BATCH_SIZE:-32}
WORKERS=${WORKERS:-8}
DET_TOKENS=${DET_TOKENS:-20}
NUM_FRAMES=${NUM_FRAMES:-9}
ANN_FILE=${ANN_FILE:-"/mnt/SSD2/coco2017/annotations/person_keypoints_val2017.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"output/eval_results"}

python val_pose.py \
    --checkpoint "$CHECKPOINT" \
    --model "$MODEL" \
    --size "$SIZE" \
    --dataset coco \
    --data_root "$COCO_ROOT" \
    --ann_file "$ANN_FILE" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --det_tokens "$DET_TOKENS" \
    --num_frames "$NUM_FRAMES" \
    --output_dir "$OUTPUT_DIR"