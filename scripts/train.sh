#!/bin/bash
cd "$(dirname "$0")/.."

# Disable Python output buffering so logs appear immediately
export PYTHONUNBUFFERED=1

# Multi-GPU training with torchrun (recommended)
# Adjust --nproc_per_node to match number of GPUs
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 train_pose.py \
       -MODEL sia_pose_simple -SIZE l14 \
       -COCO_ROOT /mnt/SSD2/coco2017/images \
       -TRAIN_ANN /mnt/SSD2/coco2017/annotations/person_keypoints_train2017.json \
       -VAL_ANN /mnt/SSD2/coco2017/annotations/person_keypoints_val2017.json \
       -BS 32 -EPOCH 50 -LR 1e-4 --SAVE -FRAMES 1 -VAL_BATCH_FREQ 100 -WORKERS 8 -LR_BACKBONE 1e-5 \
      #  --RESUME weights/avak_b16_11.pt \
       # --NO_TQDM 
       # --WANDB -WANDB_PROJECT sia-pose -WANDB_RUN pose_rle_experiment

# CUDA_VISIBLE_DEVICES=2 
