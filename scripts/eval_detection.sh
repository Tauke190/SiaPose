#!/bin/bash

# Simple detection evaluation script

set -e

CHECKPOINT="weights/avak_b16_11.pt"
DATA_ROOT="/mnt/SSD2/coco2017/images"  
MODEL="sia_pose_simple"
SIZE="b16"
BATCH_SIZE=16
DEVICE=""
OUTPUT_DIR=${7:-"output/detection_results"}



CMD="python val_detection.py \
  --checkpoint \"$CHECKPOINT\" \
  --data_root \"$DATA_ROOT\" \
  --model \"$MODEL\" \
  --size \"$SIZE\" \
  --batch_size $BATCH_SIZE"

[[ -n "$DEVICE" ]] && CMD="$CMD --device \"$DEVICE\""
[[ -n "$OUTPUT_DIR" ]] && CMD="$CMD --output_dir \"$OUTPUT_DIR\""

eval "$CMD"