#!/bin/bash
#SBATCH --job-name=sia_pose_100_tokens
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
cat $0
echo "========================================"

cd /home/av354855/SiaPose

torchrun --nproc_per_node=2 train_pose.py \
       -MODEL sia_pose_simple \
       -COCO_ROOT /home/c3-0/datasets/coco \
       -TRAIN_ANN /home/c3-0/datasets/coco/annotations/person_keypoints_train2017.json \
       -VAL_ANN /home/c3-0/datasets/coco/annotations/person_keypoints_val2017.json \
       -BS 128 -EPOCH 300 -LR 1e-4 --SAVE -FRAMES 1 \
       -WORKERS 12 -EXP sia_pose_simple -DET 100 -HEIGHT 480 -WIDTH 640 \
       --NO_TQDM --WANDB -WANDB_PROJECT sia-pose -WANDB_RUN pose_rle_experiment_100_tokens