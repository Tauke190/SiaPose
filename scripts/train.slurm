#!/bin/bash
#SBATCH --job-name=sia_pose
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48
#SBATCH --gres=gpu:1
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

cd /home/av354855/projects/sia_pose

torchrun --nproc_per_node=2 train_pose.py \
       -MODEL sia_pose_simple \
       -COCO_ROOT /mnt/SSD2/coco2017/images \
       -TRAIN_ANN /mnt/SSD2/coco2017/annotations/person_keypoints_train2017.json \
       -VAL_ANN /mnt/SSD2/coco2017/annotations/person_keypoints_val2017.json \
       --RESUME weights/avak_b16_11.pt \
       -BS 32 -EPOCH 50 -LR 1e-4 --SAVE \
       -WORKERS 8 
       # --NO_TQDM 
       # --WANDB -WANDB_PROJECT sia-pose -WANDB_RUN pose_rle_experiment
