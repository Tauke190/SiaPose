#!/bin/bash
cd "$(dirname "$0")/.."

# Disable Python output buffering so logs appear immediately
export PYTHONUNBUFFERED=1

# Multi-GPU training with torchrun (recommended)
# Adjust --nproc_per_node to match number of GPUs
torchrun --nproc_per_node=1 --master_port=29501 train_pose.py \
       -MODEL sia_pose_simple \
       -COCO_ROOT /mnt/SSD2/coco2017/images \
       -TRAIN_ANN /mnt/SSD2/coco2017/annotations/person_keypoints_train2017.json \
       -VAL_ANN /mnt/SSD2/coco2017/annotations/person_keypoints_val2017.json \
       --RESUME weights/avak_b16_11.pt \
       -BS 32 -EPOCH 50 -LR 1e-4 --SAVE \
       -WORKERS 8 
       # --NO_TQDM 
       # --WANDB -WANDB_PROJECT sia-pose -WANDB_RUN pose_rle_experiment

# CUDA_VISIBLE_DEVICES=2 