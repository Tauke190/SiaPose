#!/bin/bash
# Evaluate SIA Pose model on COCO val2017
#
# Usage:
#   bash scripts/eval.sh
#
# Override defaults with environment variables:
#   CHECKPOINT=weights/my_model.pt MODEL=sia_pose bash scripts/eval.sh

CHECKPOINT=${CHECKPOINT:-"weights/sia_pose_simple_b16_best.pt"}
MODEL=${MODEL:-"sia_pose_simple"}
SIZE=${SIZE:-"b16"}
DATASET=${DATASET:-"coco"}
BATCH_SIZE=${BATCH_SIZE:-8}
WORKERS=${WORKERS:-4}
DET_TOKENS=${DET_TOKENS:-20}
NUM_FRAMES=${NUM_FRAMES:-9}
POSE_LAYERS=${POSE_LAYERS:-3}

if [ "$DATASET" = "coco" ]; then
    DATA_ROOT=${COCO_ROOT:-"/mnt/SSD2/coco2017/images"}
    ANN_FILE=${COCO_ANN_FILE:-"/mnt/SSD2/coco2017/annotations/person_keypoints_val2017.json"}
    OUTPUT_DIR=${OUTPUT_DIR:-"output/eval_results_coco"}
elif [ "$DATASET" = "posetrack" ]; then
    DATA_ROOT=${POSETRACK_ROOT:-"/mnt/SSD2/posetrack/posetrack_2017"}
    ANN_FILE=${POSETRACK_ANN_FILE:-"/mnt/SSD2/posetrack/posetrack_2017/jsons/posetrack_val_15kpts.json"}
    OUTPUT_DIR=${OUTPUT_DIR:-"output/eval_results_posetrack"}
else
    echo "Unknown dataset: $DATASET"
    exit 1


python val_pose.py \
    --checkpoint "$CHECKPOINT" \
    --model "$MODEL" \
    --size "$SIZE" \
    --dataset "$DATASET" \
    --data_root "$DATA_ROOT" \
    --ann_file "$ANN_FILE" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --det_tokens "$DET_TOKENS" \
    --num_frames "$NUM_FRAMES" \
    --pose_layers "$POSE_LAYERS" \
    --output_dir "$OUTPUT_DIR"