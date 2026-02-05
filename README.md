# SIA Pose Estimation

An end-to-end trainable framework for **human pose estimation** built on Vision Transformers (ViT). This repository contains the implementation of the SIA pose estimation model that predicts human keypoints from video frames and images.

## Features

- **Pose Estimation**: Predicts 17 COCO keypoints from video frames and images
- **Multi-GPU Training**: Distributed training using PyTorch's `torchrun`
- **Video & Image Inference**: Scripts for inference on images and videos
- **COCO Dataset Support**: Full support for COCO keypoint detection benchmark

## Repository Structure

```
sia_pose/
├── sia/                    # Core model implementations
│   ├── sia_pose.py        # Pose estimation model with decoder
│   ├── sia_pose_simple.py # Simplified pose model (direct keypoint regression)
│   ├── sia_vision.py      # Vision Transformer backbone
│   └── simple_tokenizer.py
├── datasets/              # Dataset loading modules
│   ├── COCOPose.py       # COCO keypoint dataset
├── utils/                 # Utility functions
│   ├── config.py         # Configuration management
│   ├── optimizer.py      # Optimization utilities
│   ├── scheduler.py      # Learning rate scheduling
│   └── logger.py         # Logging utilities
├── scripts/              # Training and inference scripts
│   ├── train.sh          # Multi-GPU training script
│   ├── run_inference_image.sh
│   └── run_inference_video.sh
├── train_pose.py         # Main training script for pose estimation
├── inference_image_pose.py # Image inference with pose
├── inference_video_pose.py # Video inference with pose
└── weights/              # Pre-trained model weights
    └── pose_coco/        # COCO pose estimation weights
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+
- PyTorch 2.0+

### Setup

```bash
pip install -r extras/requirements.txt
```

**Key Dependencies:**
- torch, torchvision, torchmetrics
- pycocotools, faster-coco-eval
- opencv-contrib-python
- timm, einops, transformers
- decord (video processing)

## Quick Start

### Training Pose Estimation on COCO

```bash
bash scripts/train.sh
```

Configuration is handled via command-line arguments:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_pose.py \
    -MODEL sia_pose_simple \
    -COCO_ROOT /path/to/coco/images \
    -TRAIN_ANN /path/to/annotations/train.json \
    -VAL_ANN /path/to/annotations/val.json \
    --RESUME /path/to/pretrained.pt \
    -BS 32 -EPOCH 50 -LR 1e-4 -WORKERS 8
```

### Image Inference

```bash
python inference_image_pose.py \
    --image_path example.jpg \
    --weight_path weights/pose_coco/sia_pose_simple_b16_best.pt \
    --conf_thresh 0.5
```

### Video Inference

```bash
python inference_video_pose.py \
    --video_path example.mp4 \
    --weight_path weights/pose_coco/sia_pose_simple_b16_best.pt
```

## Models

### sia_pose_simple
Simplified pose estimation model that regresses keypoints directly from the Vision Transformer's detection tokens without a cross-attention decoder. Faster inference with competitive accuracy.

### sia_pose
Full pose estimation model with a cross-attention decoder for spatial refinement of keypoints.

### sia_pose_decoder_led
LED (Light-weight Efficient Decoder) variant with reduced parameters.

## Datasets

The framework supports the COCO keypoint detection benchmark:

- **COCO**: 17-keypoint pose estimation benchmark with ~330K images

## Key Scripts

| Script | Purpose |
|--------|---------|
| `train_pose.py` | Train pose estimation model |
| `inference_image_pose.py` | Pose inference on images |
| `inference_video_pose.py` | Pose inference on videos |

## Configuration

Models are configured through command-line arguments (see `utils/config.py`):

- `-MODEL`: Model architecture (sia_pose_simple, sia_pose, etc.)
- `-BS`: Batch size
- `-EPOCH`: Number of training epochs
- `-LR`: Learning rate
- `-WORKERS`: Number of data loading workers
- `--RESUME`: Path to resume from checkpoint
- `--SAVE`: Save best checkpoint
- `--WANDB`: Enable Weights & Biases logging

## Pre-trained Weights

- `weights/pose_coco/sia_pose_simple_b16_best.pt` - COCO pose estimation model

## Distributed Training

The training scripts use PyTorch's `DistributedDataParallel` for multi-GPU training:

```bash
torchrun --nproc_per_node=2 train_pose.py [args...]
```

Adjust `--nproc_per_node` to match your number of available GPUs.

## Performance Metrics

The framework evaluates using:
- **Pose**: OKS (Object Keypoint Similarity), AP, AR for keypoint detection

## License

See `extras/LICENSE` for details.

## References

- Vision Transformer (ViT) backbone architecture
- COCO Keypoint Detection Challenge
