#!/bin/bash
# Evaluate SIA bounding box detection on COCO val2017
#
# Usage:
#   bash scripts/eval_detection.sh
#
# Override defaults with environment variables:
#   CHECKPOINT=weights/my_model.pt bash scripts/eval_detection.sh

CHECKPOINT=${CHECKPOINT:-"weights/sia_ROIAlign_2.pt"}
MODEL=${MODEL:-"sia_pose_coco"}
SIZE=${SIZE:-"b16"}
BATCH_SIZE=${BATCH_SIZE:-8}
WORKERS=${WORKERS:-4}
DET_TOKENS=${DET_TOKENS:-20}
NUM_FRAMES=${NUM_FRAMES:-1}
POSE_LAYERS=${POSE_LAYERS:-3}
CONF_THRESH=${CONF_THRESH:-0.5}

DATA_ROOT=${COCO_ROOT:-"/mnt/SSD2/coco2017/images"}
ANN_FILE=${COCO_ANN_FILE:-"/mnt/SSD2/coco2017/annotations/person_keypoints_val2017.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"output/eval_results_coco_detection"}

python val_detection.py \
    --checkpoint "$CHECKPOINT" \
    --model "$MODEL" \
    --size "$SIZE" \
    --data_root "$DATA_ROOT" \
    --ann_file "$ANN_FILE" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --det_tokens "$DET_TOKENS" \
    --num_frames "$NUM_FRAMES" \
    --pose_layers "$POSE_LAYERS" \
    --conf_thresh "$CONF_THRESH" \
    --output_dir "$OUTPUT_DIR"
