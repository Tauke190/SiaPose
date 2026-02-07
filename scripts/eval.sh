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
POSETRACK_ROOT=${POSETRACK_ROOT:-"/mnt/SSD2/posetrack/posetrack_2017"}


BATCH_SIZE=${BATCH_SIZE:-32}
WORKERS=${WORKERS:-8}
DET_TOKENS=${DET_TOKENS:-20}
NUM_FRAMES=${NUM_FRAMES:-9}
COCO_ANN_FILE=${ANN_FILE:-"/mnt/SSD2/coco2017/annotations/person_keypoints_val2017.json"}
POSETRACK_ANN_FILE=${POSETRACK_ANN_FILE:-"/mnt/SSD2/posetrack/posetrack_2017/jsons/posetrack_val_15kpts.json"}


OUTPUT_DIR=${OUTPUT_DIR:-"output/eval_results_posetrack"}

python val_pose.py \
    --checkpoint "$CHECKPOINT" \
    --model "$MODEL" \
    --size "$SIZE" \
    --dataset posetrack \
    --data_root "$POSETRACK_ROOT" \
    --ann_file "$POSETRACK_ANN_FILE" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --det_tokens "$DET_TOKENS" \
    --num_frames "$NUM_FRAMES" \
    --output_dir "$OUTPUT_DIR"


python val_pose.py \
    --checkpoint $CHECKPOINT$ \
    --model sia_pose_simple \
    --dataset posetrack \
    --data_root /mnt/SSD2/posetrack/posetrack_2017 \
    --ann_file /mnt/SSD2/posetrack/posetrack_2017/jsons/posetrack_val_15kpts.json