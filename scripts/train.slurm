#!/bin/bash
#SBATCH --job-name=sia_pose_dec_20
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=6G 
#SBATCH --cpus-per-gpu=12


PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"

export PYTHONIOENCODING=utf-8

# Load necessary modules
module load anaconda3
module load cuda

eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate sia
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

echo "Checking PyTorch CUDA version and availability:"
python -c "import torch; print('torch.version.cuda:', torch.version.cuda); print('torch.cuda.is_available:', torch.cuda.is_available())"

echo "========================================"
echo "Training script configuration:"
echo "========================================"
cat "$(readlink -f "$0")"
echo "========================================"

cd /home/av354855/SiaPose

MODEL=sia_pose_coco_roi_best  # [sia_pose_simple, sia_pose_coco_decoder , sia_pose_coco_roi]

torchrun --nproc_per_node=2 train_pose.py \
       -MODEL $MODEL \
       -ROOT /home/c3-0/datasets/coco \
       -TRAIN_ANN /home/c3-0/datasets/coco/annotations/person_keypoints_train2017.json \
       -VAL_ANN /home/c3-0/datasets/coco/annotations/person_keypoints_val2017.json \
       -BS 64 -EPOCH 150 -LR 5e-4 --SAVE -FRAMES 1 \
       -WORKERS 12 -EXP sia_pose -DET 20 -HEIGHT 480 -WIDTH 640 -LOG 100 -POSE_LAYERS 3 -ROI_SIZE 14 -GRAD_CLIP 1\
       --RESUME weights/avak_b16_11.pt \
       --WANDB -WANDB_PROJECT sia-pose -WANDB_RUN sia_pose_coco_roialign_fusion