"""
Training script for SIA Pose Estimation on COCO Keypoints.

"""
import os
import argparse
import json
import math
import time
import numpy as np
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from sia import get_sia_pose_simple, get_sia_pose_coco_decoder, get_sia_pose_coco_roi, get_sia_pose_coco_roi_best, get_sia_pose_posetrack, get_sia_pose_eomt, HungarianMatcher, SetCriterion, PostProcessPose
from datasets import COCOPose, COCOPoseVal, PoseTrackPose, PoseTrackPoseVal
from val_utils import COCO_SIGMAS, compute_oks, box_iou_np, run_coco_eval

# PoseTrack 2017 keypoint sigmas (15 keypoints)
POSETRACK_SIGMAS = np.array([
    0.026,              # nose
    0.025, 0.025,       # head_bottom, head_top
    0.079, 0.079,       # left_shoulder, right_shoulder
    0.072, 0.072,       # left_elbow, right_elbow
    0.062, 0.062,       # left_wrist, right_wrist
    0.107, 0.107,       # left_hip, right_hip
    0.087, 0.087,       # left_knee, right_knee
    0.089, 0.089,       # left_ankle, right_ankle
])

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train SIA for Pose Estimation on COCO")

    # Model
    parser.add_argument("-MODEL", type=str, default='sia_pose_simple', choices=['sia_pose_simple', 'sia_pose_coco_decoder', 'sia_pose_coco_roi', 'sia_pose_coco_roi_best', 'sia_pose_eomt'],
                        help="Model type: sia_pose_simple / sia_pose_coco_decoder / sia_pose_coco_roi / sia_pose_coco_roi_best / sia_pose_eomt")
    parser.add_argument("-SIZE", type=str, default='b16', choices=['b16', 'l14', 'dino_b14', 'dino_l14'],
                        help="Model size: b16/l14 (ViCLIP), dino_b14/dino_l14 (DINOv2)")
    parser.add_argument("-FRAMES", type=int, default=1,
                        help="Number of input frames (image duplicated)")
    parser.add_argument("-DET", type=int, default=20,
                        help="Number of detection tokens")
    parser.add_argument("-POSE_LAYERS", type=int, default=3,
                        help="Number of pose decoder layers")
    parser.add_argument("-ROI_SIZE", type=int, default=14,
                        help="ROI output size P (each ROI -> PxP tokens via roi_align). 14=196 tokens, 7=49 tokens. 0=fallback to variable-length")
    parser.add_argument("--USE_FPN", action="store_true",
                        help="Enable ViTDet simple FPN neck: 4-level pyramid (P2-P5) from final ViT features. Routes each detection to the correct scale level based on bbox area. Use with smaller -ROI_SIZE (e.g., 7) for efficiency.")
    parser.add_argument("--FREEZE_BACKBONE", action="store_true",
                        help="Freeze vision encoder, train only pose heads (use with pretrained weights)")

    # Dataset
    parser.add_argument("-ROOT", type=str, required=True,
                        help="Path to dataset root (COCO: contains train2017/val2017/; PoseTrack: base directory)")
    parser.add_argument("-TRAIN_ANN", type=str, default=None,
                        help="Path to train keypoints annotation JSON. Default for COCO: ROOT/annotations/person_keypoints_train2017.json. Required for PoseTrack.")
    parser.add_argument("-VAL_ANN", type=str, default=None,
                        help="Path to val keypoints annotation JSON. Default for COCO: ROOT/annotations/person_keypoints_val2017.json. Required for PoseTrack.")
    parser.add_argument("-MIN_KP", type=int, default=1,
                        help="Minimum visible keypoints per person")

    # Training
    parser.add_argument("-BS", type=int, default=32,
                        help="Batch size (per GPU)")
    parser.add_argument("-WORKERS", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("-EPOCH", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("-LR", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("-LR_BACKBONE", type=float, default=2e-5,
                        help="Learning rate for backbone (lower than decoder)")
    parser.add_argument("-WEIGHT_DECAY", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("-GRAD_CLIP", type=float, default=0,
                        help="Gradient clipping max norm (0 = disabled). Reduced to 1.0 for stable training with unified tokens.")

    # Input size
    parser.add_argument("-WIDTH", type=int, default=640,
                        help="Input width")
    parser.add_argument("-HEIGHT", type=int, default=480,
                        help="Input height")

    parser.add_argument("-W_BBOX", type=float, default=2.0,
                        help="Weight for bbox L1 loss")
    parser.add_argument("-W_GIOU", type=float, default=2.0,
                        help="Weight for bbox GIoU loss")
    parser.add_argument("-W_HUMAN", type=float, default=2.0,
                        help="Weight for human classification loss")
    parser.add_argument("-W_KP", type=float, default=5.0,
                        help="Weight for keypoint RLE loss")
    parser.add_argument("-W_KP_VIS", type=float, default=2.0,
                        help="Weight for keypoint visibility loss")
    parser.add_argument("-COST_KP", type=float, default=1.0,
                        help="Keypoint cost in Hungarian matching")

    # Output
    parser.add_argument("-EXP", type=str, default='pose_coco',
                        help="Experiment name for saving")
    parser.add_argument("--SAVE", action='store_true',
                        help="Save model checkpoint every epoch")
    parser.add_argument("-BACKBONE", type=str, default=None,
                        help="Path to pretrained backbone/encoder weights to initialise from (strict=False)")
    parser.add_argument("--RESUME", type=str, default=None,
                        help="Path to a training checkpoint to continue training "
                             "(restores model weights, optimizer, scheduler, and epoch)")

    # Validation & Logging
    parser.add_argument("-VAL_FREQ", type=int, default=1,
                        help="Validate every N epochs")
    parser.add_argument("-VAL_BATCH_FREQ", type=int, default=0,
                        help="Validate every N batches within an epoch (0 = disabled)")
    parser.add_argument("--NO_VAL", action='store_true',
                        help="Skip validation")
    parser.add_argument("-LOG", type=int, default=100,
                        help="Log losses and ROI stats every N batches")

    # GPU
    parser.add_argument("-NGPU", type=int, default=None,
                        help="Number of GPUs to use (default: use all available GPUs)")

    # Weights & Biases
    parser.add_argument("--WANDB", action='store_true',
                        help="Enable Weights & Biases logging")
    parser.add_argument("-WANDB_PROJECT", type=str, default='sia-pose',
                        help="W&B project name")
    parser.add_argument("-WANDB_RUN", type=str, default=None,
                        help="W&B run name (default: EXP name)")
    parser.add_argument("-WANDB_ENTITY", type=str, default=None,
                        help="W&B entity/team name")

    args = parser.parse_args()

    # Set default annotation paths (COCO structure only)
    # PoseTrack users must provide explicit -TRAIN_ANN and -VAL_ANN
    if args.MODEL != 'sia_pose_posetrack':
        if args.TRAIN_ANN is None:
            args.TRAIN_ANN = os.path.join(args.ROOT, 'annotations', 'person_keypoints_train2017.json')
        if args.VAL_ANN is None:
            args.VAL_ANN = os.path.join(args.ROOT, 'annotations', 'person_keypoints_val2017.json')
    else:
        # PoseTrack: validate that annotation paths are provided
        if args.TRAIN_ANN is None or args.VAL_ANN is None:
            print("ERROR: PoseTrack model requires explicit -TRAIN_ANN and -VAL_ANN arguments", flush=True)
            print("  Example: -TRAIN_ANN /path/to/posetrack/train.json -VAL_ANN /path/to/posetrack/val.json", flush=True)
            sys.exit(1)

    return args


def ddp_setup():
    """Initialize DDP using torchrun environment variables."""
    if 'RANK' in os.environ:
        # Launched with torchrun
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        print(f"[Rank {rank}] DDP initialized on GPU {torch.cuda.current_device()}", flush=True)
    elif torch.cuda.is_available():
        # Single GPU mode
        torch.cuda.set_device(0)
        print(f"[Rank 0] Single GPU mode on GPU 0", flush=True)


def get_rank():
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get world size."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def collate_fn(batch):
    """Custom collate function for variable number of targets."""
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])  # Stack clips
    return tuple(batch)


def build_model(args, rank):
    """Build SIA model with pose detection."""
    if args.SIZE == 'b16':
        pretrain = "weights/ViCLIP-B_InternVid-FLT-10M.pth"
        size = 'b'
    elif args.SIZE == 'l14':
        pretrain = "weights/ViCLIP-L_InternVid-FLT-10M.pth"
        size = 'l'
    else:
        raise ValueError(f"Invalid SIZE '{args.SIZE}' for {args.MODEL}. Use 'b16' or 'l14'.")

    if not os.path.exists(pretrain):
            print(f"Warning: Pretrain weights not found at {pretrain}, training from scratch")
            pretrain = None


    if args.MODEL == 'sia_pose_simple':
        # Simplified model without decoder
        model = get_sia_pose_simple(
            size=size,
            pretrain=pretrain,
            det_token_num=args.DET,
            num_frames=args.FRAMES,
            num_keypoints=17,
        )['sia']
    elif args.MODEL == 'sia_pose_coco_decoder':
        # Lightweight decoder: pose queries cross-attend to encoder spatial features
        model = get_sia_pose_coco_decoder(
            size=size,
            pretrain=pretrain,
            det_token_num=args.DET,
            num_frames=args.FRAMES,
            num_keypoints=17,
            decoder_layers=args.POSE_LAYERS,
        )['sia']
    elif args.MODEL == 'sia_pose_coco_roi':
        # ROI-based decoder: pose queries cross-attend only to ROI features
        model = get_sia_pose_coco_roi(
            size=size,
            pretrain=pretrain,
            det_token_num=args.DET,
            num_frames=args.FRAMES,
            num_keypoints=17,
            decoder_layers=args.POSE_LAYERS,
            roi_output_size=args.ROI_SIZE,
        )['sia']
    elif args.MODEL == 'sia_pose_coco_roi_best':
        model = get_sia_pose_coco_roi_best(
            size=size,
            pretrain=pretrain,
            det_token_num=args.DET,
            num_frames=args.FRAMES,
            num_keypoints=17,
            decoder_layers=args.POSE_LAYERS,
            roi_output_size=args.ROI_SIZE,
            use_fpn=args.USE_FPN,
        )['sia']
    elif args.MODEL == 'sia_pose_posetrack':
        # ROI-based decoder: pose queries cross-attend only to ROI features
        model = get_sia_pose_posetrack(
            size=size,
            pretrain=pretrain,
            det_token_num=args.DET,
            num_frames=args.FRAMES,
            num_keypoints=15,
            decoder_layers=args.POSE_LAYERS,
            roi_output_size=args.ROI_SIZE,
        )['sia']
    elif args.MODEL == 'sia_pose_eomt':
        # EOMT: pretrained ViT split into Stage 1 (patches) + Stage 2 (patches + pose tokens)
        model = get_sia_pose_eomt(
            size=size,
            pretrain=pretrain,
            pose_token_num=args.DET,
            num_frames=args.FRAMES,
            num_keypoints=17,
            stage2_layers=args.POSE_LAYERS,
        )['sia']

    else:
        raise ValueError(f"Unknown model type: {args.MODEL}")

    return model


def build_criterion(args, rank):
    """Build matcher and criterion for training."""
    # Heatmap model: keypoint matching not applicable (global heatmaps), use bbox/human costs only
    cost_kp = 0 if args.MODEL == 'sia_pose_heatmap' else args.COST_KP

    matcher = HungarianMatcher(
        cost_class=0,  # No action classes
        cost_bbox=args.W_BBOX,
        cost_giou=args.W_GIOU,
        cost_human=args.W_HUMAN,
        cost_keypoint=cost_kp
    )

    # Include bbox and human losses so all parameters receive gradients (required for DDP)
    weight_dict = {
        'loss_keypoints': args.W_KP,
        'loss_keypoint_vis': args.W_KP_VIS,
        'loss_bbox': args.W_BBOX,
        'loss_giou': args.W_GIOU,
        'loss_human': args.W_HUMAN,
    }

    # Heatmap model uses global heatmap loss instead of per-instance RLE loss
 
    # Main loss computation
    losses_list = ['keypoints', 'boxes', 'human']

    num_kp = 15 if args.MODEL == 'sia_pose_posetrack' else 17


    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.3, # Increased from 0.1 to better suppress false positives and encourage sparse detections
        losses=losses_list,
        num_keypoints=num_kp  # COCO has 17 keypoints, needed for RLE loss sigma
    )
    criterion.to(rank)

    return criterion, weight_dict


def freeze_encoder(model):
    """Freeze encoder/backbone parameters, keeping only pose decoder trainable."""
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        # Keep pose/detection heads trainable, freeze backbone
        if any(k in name for k in [
            'pose_decoder', 'keypoint', 'pose_decoder_ln',  # pose estimation heads
            'led_decoder', 'human_embed', 'bbox_embed',                    # LED/DINOv2 heads
            'temporal_positional_embedding', 'ln_post',                    # DINOv2-specific
            'heatmap_decoder',                                             # heatmap decoder
            'pose_decoder_module', 'pose_decoder_ln',                      # lightweight pose decoder
            'roi_refine_layers', 'roi_refine_ln',                          # ROI refinement
            'roi_pos_embed', 'fpn_neck', 'fpn_roi_pos_embed',               # ROI positional embedding / ViTDet FPN
            'det_proj', 'pose_proj',                                       # Token splitting projections
            'stage2_encoder', 'pose_token', 'pose_positional_embedding',   # EOMT model
        ]):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    return frozen_count, trainable_count


def count_parameters(model):
    """Count and print trainable/frozen parameters with layer details."""
    trainable_params = 0
    frozen_params = 0

    trainable_layers = []  # List of (name, shape, numel)
    frozen_breakdown = {}

    for name, param in model.named_parameters():
        # Get module name (first part of parameter name)
        module = name.split('.')[0] if '.' in name else name

        if param.requires_grad:
            trainable_params += param.numel()
            trainable_layers.append((name, tuple(param.shape), param.numel()))
        else:
            frozen_params += param.numel()
            frozen_breakdown[module] = frozen_breakdown.get(module, 0) + param.numel()

    total_params = trainable_params + frozen_params

    print(f"\n{'='*70}")
    print(f"Parameter Summary:")
    print(f"{'='*70}")
    print(f"Total parameters:     {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({100*trainable_params/total_params:.1f}%)")
    print(f"Frozen parameters:    {frozen_params / 1e6:.2f}M ({100*frozen_params/total_params:.1f}%)")

    if trainable_layers:
        print(f"\n{'='*70}")
        print(f"Trainable Layers ({len(trainable_layers)} layers):")
        print(f"{'='*70}")
        print(f"{'Layer Name':<50} {'Shape':<20} {'Params':>10}")
        print(f"{'-'*70}")
        for name, shape, numel in trainable_layers:
            shape_str = str(list(shape))
            print(f"{name:<50} {shape_str:<20} {numel:>10,}")

    if frozen_breakdown:
        print(f"\n{'='*70}")
        print(f"Frozen Modules (summary):")
        print(f"{'='*70}")
        for module, count in sorted(frozen_breakdown.items(), key=lambda x: -x[1]):
            print(f"  {module}: {count / 1e6:.2f}M")
    print(f"{'='*70}\n")

    return trainable_params, frozen_params


def build_optimizer(model, args, criterion=None):
    """Build optimizer with separate learning rates for backbone and heads.

    Positional embeddings (positional_embedding, class_embedding, etc.) are excluded
    from weight decay via model.no_weight_decay(), following standard ViT/DETR practice.
    This prevents AdamW from shrinking slot-identity embeddings toward zero.
    """
    # Collect parameter names that should not be weight-decayed (positional embeddings etc.)
    no_decay_names = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else set()

    def is_head(name):
        return ('pose_decoder' in name or 'keypoint' in name or
                'led_decoder' in name or 'human_head' in name or
                'box_head' in name or 'det_token' in name or
                'det_positional' in name or
                'pose_decoder_ln' in name or
                'human_embed' in name or 'bbox_embed' in name or
                'temporal_positional_embedding' in name or 'ln_post' in name or
                'heatmap_decoder' in name or
                'pose_decoder_module' in name or
                'roi_refine_layers' in name or 'roi_refine_ln' in name or
                'roi_pos_embed' in name or 'fpn_neck' in name or 'fpn_roi_pos_embed' in name or
                'det_proj' in name or 'pose_proj' in name or
                'stage2_encoder' in name or 'pose_token' in name or 'pose_positional_embedding' in name)

    if args.FREEZE_BACKBONE:
        # Backbone frozen — single group, still respect no_weight_decay
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in no_decay_names:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        # Add criterion parameters (e.g., log_sigma for RLE loss)
        if criterion is not None and hasattr(criterion, 'log_sigma'):
            decay_params.append(criterion.log_sigma)
        param_groups = [
            {'params': decay_params,    'lr': args.LR, 'weight_decay': args.WEIGHT_DECAY},
            {'params': no_decay_params, 'lr': args.LR, 'weight_decay': 0.0},
        ]
    else:
        # Backbone unfrozen — 4 groups: (backbone|head) × (decay|no_decay)
        backbone_decay, backbone_no_decay = [], []
        head_decay, head_no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_no_decay = name in no_decay_names
            if is_head(name):
                (head_no_decay if is_no_decay else head_decay).append(param)
            else:
                (backbone_no_decay if is_no_decay else backbone_decay).append(param)

        # Add criterion parameters (e.g., log_sigma for RLE loss) to head_decay
        if criterion is not None and hasattr(criterion, 'log_sigma'):
            head_decay.append(criterion.log_sigma)

        param_groups = [
            {'params': backbone_decay,    'lr': args.LR_BACKBONE, 'weight_decay': args.WEIGHT_DECAY},
            {'params': backbone_no_decay, 'lr': args.LR_BACKBONE, 'weight_decay': 0.0},
            {'params': head_decay,        'lr': args.LR,          'weight_decay': args.WEIGHT_DECAY},
            {'params': head_no_decay,     'lr': args.LR,          'weight_decay': 0.0},
        ]

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer


def build_scheduler(optimizer, total_epochs, warmup_epochs=10, flat_epochs=15):
    """Build a learning rate scheduler with linear warmup + flat hold + cosine annealing.

    Args:
        optimizer: torch optimizer
        total_epochs: total number of training epochs
        warmup_epochs: number of epochs for linear warmup (default: 10)
        flat_epochs: number of epochs to hold peak LR after warmup (default: 15)

    Returns:
        scheduler object that implements warmup → flat → cosine decay
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: 0.01x to 1.0x of base LR over warmup_epochs
            return 0.01 + 0.99 * (epoch / warmup_epochs)
        elif epoch < warmup_epochs + flat_epochs:
            # Hold at peak LR for flat_epochs
            return 1.0
        else:
            # Cosine annealing: 1.0x down to 0.01x over remaining epochs
            progress = (epoch - warmup_epochs - flat_epochs) / max(1, total_epochs - warmup_epochs - flat_epochs)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, criterion, weight_dict, dataloader, optimizer, epoch, rank, args, val_loader=None, postprocess=None, val_freq_batches=100, use_wandb=False):
    """Train for one epoch with periodic validation."""
    model.train()
    criterion.train()

    total_loss = 0
    num_batches = 0
    total_train_det = 0
    batch_stats = []
    num_total_batches = len(dataloader)
    log_freq = args.LOG  # Log every N batches
    wandb_log_freq = max(1, num_total_batches // 50)  # Log to wandb ~50 times per epoch

    # Use tqdm progress bar (disabled on non-rank-0 processes in distributed training)
    iterator = enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0), file=sys.stderr))

    for batch_idx, (samples, targets) in iterator:
        optimizer.zero_grad()

        # Move to device
        samples = samples.to(rank)
        for t in targets:
            t['boxes'] = t['boxes'].to(rank)
            t['keypoints'] = t['keypoints'].to(rank)
            # Add dummy labels for criterion (all zeros = human class)
            t['labels'] = torch.zeros(len(t['boxes']), 1).to(rank)

        # Forward
        outputs = model(samples)

        # Compute loss (num_classes=1 since we only detect humans)
        loss_dict = criterion(outputs, targets, num_classes=1)
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Backward
        losses.backward()
        
        # Clip gradients and get norm
        grad_norm = 0.0
        if args.GRAD_CLIP > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.GRAD_CLIP)
        else:
            # Compute norm without clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
        optimizer.step()

        # Track loss
        loss_value = losses.item()
        total_loss += loss_value
        num_batches += 1

        # Compute grouped DET and POSE losses for logging
        with torch.no_grad():
            det_loss = sum(
                loss_dict[k].item() * weight_dict[k]
                for k in ('loss_bbox', 'loss_giou', 'loss_human')
                if k in loss_dict and k in weight_dict
            )
            pose_loss = sum(
                loss_dict[k].item() * weight_dict[k]
                for k in ('loss_keypoints', 'loss_keypoint_vis')
                if k in loss_dict and k in weight_dict
            )

        # Count detections
        if postprocess is not None:
            with torch.no_grad():
                imgsize = (args.HEIGHT, args.WIDTH)
                batch_results = postprocess(outputs, imgsize, human_conf=0.5, keypoint_conf=0.3)
                batch_det = sum(len(r['boxes']) for r in batch_results)
                total_train_det += batch_det

        # Log progress
        if rank == 0:
            loss_str = f"loss: {loss_value:.3f} | DET: {det_loss:.3f} | POSE: {pose_loss:.3f} | grad_norm: {grad_norm:.4f}"

            # tqdm updates automatically with postfix set below

            # Log to wandb periodically
            if use_wandb and (batch_idx + 1) % wandb_log_freq == 0:
                pass  # Batch-level training logging disabled

            # Log detailed losses to console every LOG steps
            if (batch_idx + 1) % log_freq == 0 or batch_idx == 0:
                log_msg = f"Epoch {epoch} [{batch_idx + 1}/{num_total_batches}] "
                log_msg += f"loss: {loss_value:.4f} | DET: {det_loss:.4f} | POSE: {pose_loss:.4f} | grad_norm: {grad_norm:.4f}"
                for k, v in loss_dict.items():
                    if k in weight_dict:
                        log_msg += f" | {k}: {v.item():.4f}"

                # Monitor RLE sigma for divergence detection
                if hasattr(criterion, 'log_sigma'):
                    log_sigma = criterion.log_sigma.detach().cpu()
                    sigma_max = log_sigma.max().item()
                    sigma_mean = log_sigma.mean().item()
                    log_msg += f" | sigma_max: {sigma_max:.3f} sigma_mean: {sigma_mean:.3f}"

                    # Warn if sigma grows too large (divergence risk)
                    if sigma_max > 2.0:
                        log_msg += f" [WARNING: High sigma detected]"

                if rank == 0:
                    print(log_msg, flush=True)

        # Update tqdm postfix with current loss (rank 0 only)
        if rank == 0:
            try:
                iterator._iterable.set_postfix_str(loss_str[:80])
            except (AttributeError, TypeError):
                pass

        # Validate every N batches
        if val_loader is not None and val_freq_batches > 0 and (batch_idx + 1) % val_freq_batches == 0:
            if rank == 0:
                print(f"\n  Validating at batch {batch_idx + 1}...", flush=True)
            val_stats = validate(model, val_loader, postprocess, rank, args, criterion, weight_dict)
            batch_stats.append({
                'epoch': epoch,
                'batch': batch_idx + 1,
                'val_stats': val_stats
            })
            # Log validation metrics to wandb
            if use_wandb:
                global_step = epoch * num_total_batches + batch_idx
                wandb_val = {
                    'val/loss': val_stats['val_loss'],
                    'val/total_detections': val_stats['total_det'],
                    'val/total_gt': val_stats['total_gt'],
                }
                # Log AP metrics (overall, medium, large, and OKS)
                for k in ['AP', 'AP50', 'AP75', 'AP_M', 'AP_L', 'AR', 'AR_M', 'AR_L', 'mean_oks']:
                    if k in val_stats:
                        wandb_val[f'val/{k}'] = val_stats[k]
                
                # Log per-component loss breakdown
                if 'per_component_losses' in val_stats and val_stats['per_component_losses']:
                    for loss_name, loss_val in val_stats['per_component_losses'].items():
                        wandb_val[f'val/{loss_name}'] = loss_val
                
                # Log RLE sigma statistics
                if 'sigma_stats' in val_stats and val_stats['sigma_stats']:
                    for stat_name, stat_val in val_stats['sigma_stats'].items():
                        wandb_val[f'val/{stat_name}'] = stat_val
                
                wandb.log(wandb_val, step=global_step)
            model.train()  # Switch back to train mode

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, batch_stats, total_train_det


@torch.no_grad()
def validate(model, dataloader, postprocess, rank, args, criterion=None, weight_dict=None):
    """Validate the model using official COCO evaluation (pycocotools) or PoseTrack OKS.

    Wraps val_utils.validate_pose with DDP-aware dataset-specific evaluation.
    Aggregates results from all GPUs when running in distributed mode.
    """
    from val_utils import validate_pose

    model.eval()
    imgsize = (args.HEIGHT, args.WIDTH)
    world_size = get_world_size()

    # Select keypoints and sigmas based on model type
    is_posetrack = args.MODEL == 'sia_pose_posetrack'
    num_keypoints = 15 if is_posetrack else 17
    sigmas = POSETRACK_SIGMAS if is_posetrack else COCO_SIGMAS

    # Call the shared validation logic (each GPU processes its subset)
    val_results = validate_pose(
        model=model,
        dataloader=dataloader,
        postprocess=postprocess,
        imgsize=imgsize,
        num_keypoints=num_keypoints,
        sigmas=sigmas,
        criterion=criterion,
    )

    coco_results = val_results['coco_results']
    num_images = val_results['num_images']
    total_gt = val_results['total_gt']
    total_det = val_results['total_det']
    total_oks = val_results.get('total_oks', val_results['mean_oks'] * val_results['num_oks_matched'])
    val_loss_sum = val_results['val_loss'] * val_results.get('num_val_batches', max(num_images // args.BS, 1))
    num_oks_matched = val_results['num_oks_matched']
    num_val_batches = val_results.get('num_val_batches', max(num_images // args.BS, 1))
    per_component_losses = val_results.get('per_component_losses', {})
    sigma_stats = val_results.get('sigma_stats', {})

    # Aggregate stats across all GPUs in distributed mode
    if world_size > 1 and dist.is_initialized():
        device = torch.device(f'cuda:{rank}')
        
        # Aggregate scalar stats using all_reduce (sum)
        stats_tensor = torch.tensor([
            float(num_images),
            float(total_gt),
            float(total_det),
            float(total_oks),
            float(num_oks_matched),
            float(val_loss_sum),
            float(num_val_batches),
        ], device=device)
        
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        
        num_images = int(stats_tensor[0].item())
        total_gt = int(stats_tensor[1].item())
        total_det = int(stats_tensor[2].item())
        total_oks = stats_tensor[3].item()
        num_oks_matched = int(stats_tensor[4].item())
        val_loss_sum = stats_tensor[5].item()
        num_val_batches = int(stats_tensor[6].item())
        
        # Gather per-component losses from all GPUs and average them
        if per_component_losses:
            # Convert to tensor for aggregation
            component_names = sorted(per_component_losses.keys())
            component_vals = torch.tensor(
                [per_component_losses.get(k, 0.0) for k in component_names],
                dtype=torch.float32,
                device=device
            )
            dist.all_reduce(component_vals, op=dist.ReduceOp.SUM)
            # Average across GPUs
            component_vals = component_vals / world_size
            per_component_losses = {k: component_vals[i].item() for i, k in enumerate(component_names)}
        
        # Average sigma stats across GPUs
        if sigma_stats and 'sigma_mean' in sigma_stats:
            sigma_tensor = torch.tensor(
                [sigma_stats.get('sigma_mean', 0.0), 
                 sigma_stats.get('sigma_min', float('inf')), 
                 sigma_stats.get('sigma_max', float('-inf'))],
                dtype=torch.float32,
                device=device
            )
            dist.all_reduce(sigma_tensor, op=dist.ReduceOp.SUM)
            sigma_stats = {
                'sigma_mean': (sigma_tensor[0] / world_size).item(),
                'sigma_min': sigma_tensor[1].item(),
                'sigma_max': sigma_tensor[2].item(),
            }
        
        # Gather coco_results from all GPUs for COCO evaluation
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, coco_results)
        
        # Flatten gathered results (only needed on rank 0 for COCO eval)
        coco_results = []
        for results_from_gpu in gathered_results:
            coco_results.extend(results_from_gpu)

    # Compute aggregated metrics
    mean_oks = total_oks / max(num_oks_matched, 1)
    val_loss = val_loss_sum / max(num_val_batches, 1)

    # Run official COCO evaluation (only on rank 0, and only for COCO dataset)
    coco_stats = {}
    if rank == 0:
        print(f"Validation: {num_images} images, {total_gt} GT persons, {total_det} detections, "
              f"val_loss: {val_loss:.4f}, mean_oks (all sizes): {mean_oks:.4f} ({num_oks_matched} matched)", flush=True)
        # COCO evaluation via pycocotools (PoseTrack uses OKS metric only)
        if not is_posetrack:
            coco_stats = run_coco_eval(args.VAL_ANN, coco_results, sigmas=COCO_SIGMAS)

    return {
        'total_gt': total_gt,
        'total_det': total_det,
        'val_loss': val_loss,
        'mean_oks': mean_oks,
        'per_component_losses': per_component_losses,
        'sigma_stats': sigma_stats,
        **coco_stats,
    }


def main(args):
    """Main training function."""
    ddp_setup()

    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize Weights & Biases (only on rank 0)
    use_wandb = args.WANDB and WANDB_AVAILABLE and is_main_process()
    if use_wandb:
        wandb_run_name = args.WANDB_RUN if args.WANDB_RUN else args.EXP
        wandb.init(
            project=args.WANDB_PROJECT,
            entity=args.WANDB_ENTITY,
            name=wandb_run_name,
            config={
                'model': args.MODEL,
                'size': args.SIZE,
                'frames': args.FRAMES,
                'det_tokens': args.DET,
                'pose_layers': args.POSE_LAYERS,
                'batch_size': args.BS,
                'effective_batch_size': args.BS * world_size,
                'epochs': args.EPOCH,
                'lr': args.LR,
                'lr_backbone': args.LR_BACKBONE,
                'weight_decay': args.WEIGHT_DECAY,
                'w_kp': args.W_KP,
                'w_kp_vis': args.W_KP_VIS,
                'input_size': f"{args.WIDTH}x{args.HEIGHT}",
                'world_size': world_size,
            }
        )
        print(f"W&B initialized: {wandb.run.url}", flush=True)
    elif args.WANDB and not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Run 'pip install wandb' to enable.", flush=True)

    # Build model
    if is_main_process():
        print("Building model...", flush=True)
    model = build_model(args, rank)

    # Load pretrained backbone weights (strict=False: new heads are randomly initialised)
    start_epoch = 0
    _resume_train_ckpt = None  # held until optimizer/scheduler are built
    _resume_best_ap = 0.0
    if args.BACKBONE:
        if is_main_process():
            print(f"Loading pretrained backbone from {args.BACKBONE}", flush=True)
        state_dict = torch.load(args.BACKBONE, map_location='cpu', weights_only=False)

        # Handle temporal positional embedding mismatch (e.g. checkpoint=9 frames, model=1 frame)
        key = "vision_encoder.temporal_positional_embedding"
        if key in state_dict:
            old_shape = state_dict[key].shape  # [1, T_old, C]
            new_shape = model.state_dict()[key].shape  # [1, T_new, C]
            if old_shape[1] != new_shape[1]:
                if is_main_process():
                    print(f"  Interpolating temporal embed: {old_shape[1]} -> {new_shape[1]} frames", flush=True)
                old_embed = state_dict[key].unsqueeze(2).permute(0, 3, 1, 2).float()  # [1, C, T_old, 1]
                new_embed = F.interpolate(old_embed, size=(new_shape[1], 1), mode='bilinear', align_corners=False)
                state_dict[key] = new_embed.permute(0, 2, 3, 1).squeeze(2)  # [1, T_new, C]

        model.load_state_dict(state_dict, strict=False)

    # Continue training from a saved checkpoint (model + optimizer + scheduler + epoch)
    if args.RESUME:
        if is_main_process():
            print(f"Continuing training from {args.RESUME}", flush=True)
        _resume_train_ckpt = torch.load(args.RESUME, map_location='cpu', weights_only=False)
        is_full_train_ckpt = (
            isinstance(_resume_train_ckpt, dict)
            and 'model' in _resume_train_ckpt
            and 'optimizer' in _resume_train_ckpt
            and 'scheduler' in _resume_train_ckpt
            and 'epoch' in _resume_train_ckpt
        )

        if is_full_train_ckpt:
            incompatible = model.load_state_dict(_resume_train_ckpt['model'], strict=False)
            if is_main_process():
                if incompatible.missing_keys:
                    print(f"  Missing keys (will reinit): {incompatible.missing_keys}", flush=True)
                if incompatible.unexpected_keys:
                    print(f"  Unexpected keys (ignored): {incompatible.unexpected_keys}", flush=True)
            start_epoch = _resume_train_ckpt['epoch'] + 1
            _resume_best_ap = _resume_train_ckpt.get('best_ap', 0.0)
            if is_main_process():
                print(f"  Resuming from epoch {start_epoch}, best_ap={_resume_best_ap:.4f}", flush=True)
        else:
            state_dict = _resume_train_ckpt.get('model', _resume_train_ckpt) if isinstance(_resume_train_ckpt, dict) else _resume_train_ckpt
            model.load_state_dict(state_dict, strict=False)
            start_epoch = (_resume_train_ckpt.get('epoch', -1) + 1) if isinstance(_resume_train_ckpt, dict) else 0
            _resume_best_ap = _resume_train_ckpt.get('best_ap', 0.0) if isinstance(_resume_train_ckpt, dict) else 0.0
            _resume_train_ckpt = None
            if is_main_process():
                print(
                    "  Loaded weights-only checkpoint; optimizer/scheduler state not found. "
                    f"Starting with fresh optimizer/scheduler from epoch {start_epoch}.",
                    flush=True,
                )

    # Freeze encoder parameters if requested
    if args.FREEZE_BACKBONE:
        if is_main_process():
            print("Freezing backbone (vision encoder)...", flush=True)
        freeze_encoder(model)
    else:
        if is_main_process():
            print("All parameters trainable (backbone unfrozen)", flush=True)

    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Print model info
    if is_main_process():
        # Use the underlying model for DDP
        base_model = model.module if hasattr(model, 'module') else model
        count_parameters(base_model)

    # Build criterion
    criterion, weight_dict = build_criterion(args, device)

    # Build optimizer (pass criterion so log_sigma gets included)
    optimizer = build_optimizer(model, args, criterion=criterion)

    # Build scheduler with linear warmup (10 epochs) + flat hold (15 epochs) + cosine annealing
    scheduler = build_scheduler(optimizer, total_epochs=args.EPOCH, warmup_epochs=10, flat_epochs=15)

    # Restore optimizer + scheduler state when continuing a training run
    if _resume_train_ckpt is not None:
        saved_groups = len(_resume_train_ckpt['optimizer']['param_groups'])
        current_groups = len(optimizer.param_groups)
        if saved_groups != current_groups:
            if is_main_process():
                print(
                    f"  WARNING: checkpoint optimizer has {saved_groups} param groups, "
                    f"current optimizer has {current_groups}. "
                    "Skipping optimizer/scheduler restore — starting with fresh optimizer state.",
                    flush=True,
                )
        else:
            optimizer.load_state_dict(_resume_train_ckpt['optimizer'])
            scheduler.load_state_dict(_resume_train_ckpt['scheduler'])
        # Move optimizer state tensors to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        del _resume_train_ckpt  # free memory

    # Build datasets
    if is_main_process():
        print("Loading datasets...", flush=True)

    transforms = v2.Compose([
        v2.Resize((args.HEIGHT, args.WIDTH)),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Select dataset class based on model
    is_posetrack = args.MODEL == 'sia_pose_posetrack'

    if is_posetrack:
        train_dataset = PoseTrackPose(
            root=args.ROOT,  # PoseTrack base directory
            annFile=args.TRAIN_ANN,
            transforms=transforms,
            frames=args.FRAMES,
        )
    else:
        train_dataset = COCOPose(
            root=os.path.join(args.ROOT, 'train2017'),
            annFile=args.TRAIN_ANN,
            transforms=transforms,
            frames=args.FRAMES,
            min_keypoints=args.MIN_KP,
            augment=True,  # Enable coordinate-aware augmentation
        )

    # Use DistributedSampler only for multi-GPU
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.BS,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=args.WORKERS,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.BS,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.WORKERS,
            pin_memory=True,
            drop_last=True
        )

    # Validation dataset
    val_loader = None
    postprocess = None
    if not args.NO_VAL:
        if is_posetrack:
            val_dataset = PoseTrackPoseVal(
                root=args.ROOT,  # PoseTrack base directory
                annFile=args.VAL_ANN,
                transforms=transforms,
                frames=args.FRAMES,
            )
        else:
            val_dataset = COCOPoseVal(
                root=os.path.join(args.ROOT, 'val2017'),
                annFile=args.VAL_ANN,
                transforms=transforms,
                frames=args.FRAMES,
                min_keypoints=1  # Include more for validation
            )
        if world_size > 1:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.BS,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=args.WORKERS,
                pin_memory=True
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.BS,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=args.WORKERS,
                pin_memory=True
            )
        postprocess = PostProcessPose()
        

    # Create experiment directory
    exp_dir = os.path.join('output', args.EXP)
    if is_main_process() and not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    # Training loop
    if is_main_process():
        print(f"\n" + "="*80)
        print(f"DATASET SUMMARY")
        print(f"="*80)
        print(f"Training dataset:   {len(train_dataset):>7,} images")
        if hasattr(train_dataset, 'coco'):
            # Count annotations with keypoints >= 1 in the training images
            train_anns_with_kp = 0
            train_anns_total = 0
            for img_id in train_dataset.img_ids:
                ann_ids = train_dataset.coco.getAnnIds(imgIds=img_id, catIds=train_dataset.cat_ids, iscrowd=False)
                anns = train_dataset.coco.loadAnns(ann_ids)
                train_anns_total += len(anns)
                train_anns_with_kp += len([a for a in anns if a.get('num_keypoints', 0) >= 1])
            print(f"                    {train_anns_with_kp:>7,} annotations with ≥1 keypoint")
            print(f"                    {train_anns_total:>7,} total annotations")
        if not args.NO_VAL:
            print(f"Validation dataset: {len(val_dataset):>7,} images")
            if hasattr(val_dataset, 'coco'):
                # Count annotations with keypoints >= 1 in the validation images
                val_anns_with_kp = 0
                val_anns_total = 0
                for img_id in val_dataset.img_ids:
                    ann_ids = val_dataset.coco.getAnnIds(imgIds=img_id, catIds=val_dataset.cat_ids, iscrowd=False)
                    anns = val_dataset.coco.loadAnns(ann_ids)
                    val_anns_total += len(anns)
                    val_anns_with_kp += len([a for a in anns if a.get('num_keypoints', 0) >= 1])
                print(f"                    {val_anns_with_kp:>7,} annotations with ≥1 keypoint")
                print(f"                    {val_anns_total:>7,} total annotations")
        else:
            print(f"Validation dataset: DISABLED")
        print(f"Batch size: {args.BS} x {world_size} GPUs = {args.BS * world_size} effective")
        print(f"\nStarting training for {args.EPOCH} epochs...")
        print(f"="*80 + "\n", flush=True)

    best_loss = float('inf')
    best_ap = _resume_best_ap  # Restored from checkpoint, or 0.0 for a fresh run
    train_stats = []

    for epoch in range(start_epoch, args.EPOCH):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Track epoch time
        epoch_start_time = time.time()

        # Train
        avg_loss, batch_stats, total_train_det = train_one_epoch(
            model, criterion, weight_dict, train_loader, optimizer, epoch, rank, args,
            val_loader=val_loader if not args.NO_VAL else None,
            postprocess=postprocess,
            val_freq_batches=args.VAL_BATCH_FREQ,
            use_wandb=use_wandb
        )

        scheduler.step()

        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        hours = int(epoch_duration // 3600)
        minutes = int((epoch_duration % 3600) // 60)
        seconds = epoch_duration % 60
        time_str = f"{hours}h {minutes}m {seconds:.1f}s" if hours > 0 else f"{minutes}m {seconds:.1f}s"

        if is_main_process():
            print(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}, lr = {scheduler.get_last_lr()[0]:.2e}, train_det: {total_train_det}, time: {time_str}", flush=True)
            if batch_stats:
                print(f"  Batch validations: {len(batch_stats)} checkpoints", flush=True)

        # Validate
        val_stats = None
        if not args.NO_VAL and (epoch + 1) % args.VAL_FREQ == 0:
            val_stats = validate(model, val_loader, postprocess, rank, args, criterion, weight_dict)
            if is_main_process():
                train_stats.append({
                    'epoch': epoch,
                    'train_loss': avg_loss,
                    'val_stats': val_stats,
                    'batch_validations': batch_stats
                })

        # Log epoch-level metrics to wandb
        if use_wandb:
            epoch_metrics = {
                'epoch/train_loss': avg_loss,
                'epoch/lr': scheduler.get_last_lr()[0],
                'epoch': epoch,
            }
            if val_stats is not None:
                epoch_metrics['epoch/val_loss'] = val_stats['val_loss']
                # Log AP metrics (overall, medium, large, and OKS)
                for k in ['AP', 'AP50', 'AP75', 'AP_M', 'AP_L', 'AR', 'AR_M', 'AR_L', 'mean_oks']:
                    if k in val_stats:
                        epoch_metrics[f'epoch/{k}'] = val_stats[k]
                
                # Log per-component loss breakdown at epoch level
                if 'per_component_losses' in val_stats and val_stats['per_component_losses']:
                    for loss_name, loss_val in val_stats['per_component_losses'].items():
                        epoch_metrics[f'epoch/{loss_name}'] = loss_val
                
                # Log RLE sigma statistics at epoch level
                if 'sigma_stats' in val_stats and val_stats['sigma_stats']:
                    for stat_name, stat_val in val_stats['sigma_stats'].items():
                        epoch_metrics[f'epoch/{stat_name}'] = stat_val
            
            wandb.log(epoch_metrics)

        # Save checkpoint every epoch (full training state for resuming)
        if args.SAVE and is_main_process():
            model_sd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

            # Save best checkpoint as full training state package.
            if val_stats is not None and 'AP' in val_stats and val_stats['AP'] > best_ap:
                best_ap = val_stats['AP']
                best_path = os.path.join(exp_dir, f'{args.MODEL}_{args.SIZE}_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model': model_sd,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_ap': best_ap,
                }, best_path)
                print(f"Saved best checkpoint (AP: {best_ap:.4f}): {best_path}", flush=True)

            ckpt_path = os.path.join(exp_dir, f'{args.MODEL}_{args.SIZE}_checkpoint.pt')
            torch.save({
                'epoch': epoch,
                'model': model_sd,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_ap': best_ap,
            }, ckpt_path)
            ap_str = f", AP: {val_stats['AP']:.4f} (best: {best_ap:.4f})" if val_stats and 'AP' in val_stats else ""
            print(f"Saved checkpoint (epoch {epoch}{ap_str}): {ckpt_path}", flush=True)

    # Save training stats only (best model already saved during training)
    if args.SAVE and is_main_process():
        stats_path = os.path.join(exp_dir, 'train_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(train_stats, f, indent=2)
        print(f"Saved training stats: {stats_path}", flush=True)

    # Finish wandb run
    if use_wandb:
        wandb.finish()

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()

    
    # Print info
    print(f"Experiment: {args.EXP}")
    print(f"Model: {args.MODEL} ({args.SIZE})")
    print(f"Dataset root: {args.ROOT}")
    
    # Check if launched with torchrun
    if 'RANK' in os.environ:
        print(f"Launched with torchrun: WORLD_SIZE={os.environ.get('WORLD_SIZE', 1)}")
    else:
        ngpu = args.NGPU if args.NGPU is not None else torch.cuda.device_count()
        if ngpu == 0:
            print("No GPU found, using CPU (not recommended for training)")
        elif ngpu == 1:
            print("Single GPU mode")
        else:
            print(f"For multi-GPU training, use: torchrun --nproc_per_node={ngpu} train_pose.py [args]")
    
    main(args)
