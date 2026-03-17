#!/bin/bash
cd "$(dirname "$0")/.."

# Disable Python output buffering so logs appear immediately
export PYTHONUNBUFFERED=1

# Create training_logs directory if it doesn't exist
mkdir -p training_logs

# Generate log filename with timestamp
LOG_FILE="training_logs/train_$(date +%Y%m%d_%H%M%S).log"

export PYTHONIOENCODING=utf-8

MODEL=sia_pose_coco_roi_best  # [sia_pose_simple, sia_pose_coco_decoder , sia_pose_coco_roi, sia_pose_coco_roi_best]
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_pose.py \
       -MODEL $MODEL -SIZE b16 \
       -ROOT /mnt/SSD2/coco2017 \
       -TRAIN_ANN /mnt/SSD2/coco2017/annotations/person_keypoints_train2017.json \
       -VAL_ANN /mnt/SSD2/coco2017/annotations/person_keypoints_val2017.json \
       -BS 32 -EPOCH 150 -LR 5e-4 --SAVE -FRAMES 1 -VAL_BATCH_FREQ 100 -LOG 5 -WORKERS 12 -LR_BACKBONE 1e-5 -DET 20 -GRAD_CLIP 1 \
       -HEIGHT 480 -WIDTH 640 -FUSION_LAYERS 6,8,10 \
       --RESUME weights/avak_b16_11.pt \
       -POSE_LAYERS 3 -ROI_SIZE 14 2>&1 | tee "$LOG_FILE"
       # --WANDB -WANDB_PROJECT sia-pose -WANDB_RUN pose_rle_experiment


# # PoseTrack
# CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 train_pose.py \
#        -MODEL $MODEL -SIZE b16 \
#        -ROOT /home/c3-0/datasets/posetrack/posetrack_2017 \
#        -TRAIN_ANN /home/c3-0/datasets/posetrack/posetrack_2017/jsons/posetrack_train_fixed.json \
#        -VAL_ANN home/c3-0/datasets/posetrack/posetrack_2017/jsons/posetrack_val_15kpts.json \
#        -BS 32 -EPOCH 300 -LR 5e-4 --SAVE -FRAMES 9 -VAL_BATCH_FREQ 100 -LOG 100 -WORKERS 12 -LR_BACKBONE 1e-5 -DET 20 \
#        -HEIGHT 480 -WIDTH 640 \
#        --RESUME weights/avak_b16_11.pt \
#        -POSE_LAYERS 3 -ROI_SIZE 14 2>&1 | tee "$LOG_FILE"
#        # --WANDB -WANDB_PROJECT sia-pose -WANDB_RUN pose_rle_experiment