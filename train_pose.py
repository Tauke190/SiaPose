"""
Training script for SIA Pose Estimation on COCO Keypoints.

"""
import os
import argparse
import json
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

from sia import get_sia_pose, get_sia_pose_simple, get_sia_pose_dino, get_sia_pose_dino_simple, get_sia_pose_heatmap, HungarianMatcher, SetCriterion, PostProcessPose
from datasets import COCOPose, COCOPoseVal

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# COCO keypoint sigmas for OKS computation (17 keypoints)
# Order: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
COCO_SIGMAS = np.array([
    0.026,  # nose
    0.025, 0.025,  # left_eye, right_eye
    0.035, 0.035,  # left_ear, right_ear
    0.079, 0.079,  # left_shoulder, right_shoulder
    0.072, 0.072,  # left_elbow, right_elbow
    0.062, 0.062,  # left_wrist, right_wrist
    0.107, 0.107,  # left_hip, right_hip
    0.087, 0.087,  # left_knee, right_knee
    0.089, 0.089,  # left_ankle, right_ankle
])


def compute_oks(pred_xy, gt_kps, gt_area, sigmas):
    """Compute OKS between one predicted and one GT person instance.

    Args:
        pred_xy: np.array [K, 2] predicted (x, y) in original image coords
        gt_kps:  np.array [K, 3] ground truth (x, y, v) in original image coords
        gt_area: float, area of GT bounding box in pixels
        sigmas:  np.array [K], per-keypoint sigma
    Returns:
        float OKS score in [0, 1]
    """
    xg, yg, vg = gt_kps[:, 0], gt_kps[:, 1], gt_kps[:, 2]
    xd, yd = pred_xy[:, 0], pred_xy[:, 1]
    visible = vg > 0
    if visible.sum() == 0:
        return 0.0
    dx = xd - xg
    dy = yd - yg
    vars = (sigmas * 2) ** 2
    e = (dx**2 + dy**2) / (2 * gt_area * vars + np.spacing(1))
    return float(np.sum(np.exp(-e[visible])) / visible.sum())


def box_iou_np(box_a, boxes_b):
    """Compute IoU between one box and an array of boxes (xyxy format).

    Args:
        box_a:  np.array [4] (x1, y1, x2, y2)
        boxes_b: np.array [M, 4]
    Returns:
        np.array [M] of IoU values
    """
    xa1 = np.maximum(box_a[0], boxes_b[:, 0])
    ya1 = np.maximum(box_a[1], boxes_b[:, 1])
    xa2 = np.minimum(box_a[2], boxes_b[:, 2])
    ya2 = np.minimum(box_a[3], boxes_b[:, 3])
    inter = np.maximum(xa2 - xa1, 0) * np.maximum(ya2 - ya1, 0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    return inter / (area_a + area_b - inter + np.spacing(1))


def run_coco_eval(ann_file, coco_results, sigmas=None):
    """Run the official COCO keypoint evaluation via pycocotools.

    Returns:
        stats dict with AP, AP50, AP75, AR, etc.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if len(coco_results) == 0:
        print("No predictions to evaluate.")
        return {}

    res_path = '/tmp/sia_pose_train_coco_results.json'
    with open(res_path, 'w') as f:
        json.dump(coco_results, f)

    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(res_path)

    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    if sigmas is not None:
        coco_eval.params.kpt_oks_sigmas = sigmas
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    names = ['AP', 'AP50', 'AP75', 'AP_M', 'AP_L', 'AR', 'AR50', 'AR75',
             'AR_M', 'AR_L']
    stats = {n: float(v) for n, v in zip(names, coco_eval.stats)}
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Train SIA for Pose Estimation on COCO")

    # Model
    parser.add_argument("-MODEL", type=str, default='pose', choices=['sia_pose', 'sia_pose_simple', 'sia_pose_heatmap', 'sia_pose_dino', 'sia_pose_dino_simple'],
                        help="Model type: 'sia_pose' (det tokens in encoder), 'sia_pose_simple' (no decoder), 'sia_pose_decoder_led' (DETR-style LED), 'sia_pose_dino' (DINOv2 + LED), 'sia_pose_dino_simple' (DINOv2 encoder-only)")
    parser.add_argument("-SIZE", type=str, default='b16', choices=['b16', 'l14', 'dino_b14', 'dino_l14'],
                        help="Model size: b16/l14 (ViCLIP), dino_b14/dino_l14 (DINOv2)")
    parser.add_argument("-FRAMES", type=int, default=1,
                        help="Number of input frames (image duplicated)")
    parser.add_argument("-DET", type=int, default=20,
                        help="Number of detection tokens")
    parser.add_argument("-POSE_LAYERS", type=int, default=2,
                        help="Number of pose decoder layers")
    parser.add_argument("--FREEZE_BACKBONE", action="store_true",
                        help="Freeze vision encoder, train only pose heads (use with pretrained weights)")

    # Dataset
    parser.add_argument("-COCO_ROOT", type=str, required=True,
                        help="Path to COCO images root (contains train2017/, val2017/)")
    parser.add_argument("-TRAIN_ANN", type=str, default=None,
                        help="Path to train keypoints annotation JSON (default: COCO_ROOT/annotations/person_keypoints_train2017.json)")
    parser.add_argument("-VAL_ANN", type=str, default=None,
                        help="Path to val keypoints annotation JSON (default: COCO_ROOT/annotations/person_keypoints_val2017.json)")
    parser.add_argument("-MIN_KP", type=int, default=5,
                        help="Minimum visible keypoints per person")

    # Training
    parser.add_argument("-BS", type=int, default=32,
                        help="Batch size (per GPU)")
    parser.add_argument("-WORKERS", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("-EPOCH", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("-LR", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("-LR_BACKBONE", type=float, default=1e-5,
                        help="Learning rate for backbone (lower than decoder)")
    parser.add_argument("-WEIGHT_DECAY", type=float, default=1e-4,
                        help="Weight decay")

    # Input size
    parser.add_argument("-WIDTH", type=int, default=320,
                        help="Input width")
    parser.add_argument("-HEIGHT", type=int, default=240,
                        help="Input height")

    # Loss weights
    parser.add_argument("-W_BBOX", type=float, default=5.0,
                        help="Weight for bbox L1 loss")
    parser.add_argument("-W_GIOU", type=float, default=2.0,
                        help="Weight for bbox GIoU loss")
    parser.add_argument("-W_HUMAN", type=float, default=2.0,
                        help="Weight for human classification loss")
    parser.add_argument("-W_KP", type=float, default=3.0,
                        help="Weight for keypoint RLE loss")
    parser.add_argument("-W_KP_VIS", type=float, default=1.0,
                        help="Weight for keypoint visibility loss")
    parser.add_argument("-COST_KP", type=float, default=1.0,
                        help="Keypoint cost in Hungarian matching")

    # Output
    parser.add_argument("-EXP", type=str, default='pose_coco',
                        help="Experiment name for saving")
    parser.add_argument("--SAVE", action='store_true',
                        help="Save model checkpoints")
    parser.add_argument("-SAVE_FREQ", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--RESUME", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Validation
    parser.add_argument("-VAL_FREQ", type=int, default=1,
                        help="Validate every N epochs")
    parser.add_argument("-VAL_BATCH_FREQ", type=int, default=0,
                        help="Validate every N batches within an epoch (0 = disabled)")
    parser.add_argument("--NO_VAL", action='store_true',
                        help="Skip validation")
    parser.add_argument("--NO_TQDM", action='store_true',
                        help="Disable tqdm progress bar (useful for distributed training logs)")

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

    # Set default annotation paths
    if args.TRAIN_ANN is None:
        args.TRAIN_ANN = os.path.join(args.COCO_ROOT, 'annotations', 'person_keypoints_train2017.json')
    if args.VAL_ANN is None:
        args.VAL_ANN = os.path.join(args.COCO_ROOT, 'annotations', 'person_keypoints_val2017.json')

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
    # DINOv2 models don't need ViCLIP pretrain weights
    if args.MODEL == 'sia_pose_dino':
        if args.SIZE == 'dino_b14':
            dino_size = 'b'
        elif args.SIZE == 'dino_l14':
            dino_size = 'l'
        else:
            raise ValueError(f"Invalid SIZE '{args.SIZE}' for sia_pose_dino. Use 'dino_b14' or 'dino_l14'.")
        model = get_sia_pose_dino(
            size=dino_size,
            det_token_num=args.DET,
            num_frames=args.FRAMES,
            num_keypoints=17,
            decoder_layers=args.POSE_LAYERS,
        )['sia']
        return model

    if args.MODEL == 'sia_pose_dino_simple':
        if args.SIZE == 'dino_b14':
            dino_size = 'b'
        elif args.SIZE == 'dino_l14':
            dino_size = 'l'
        else:
            raise ValueError(f"Invalid SIZE '{args.SIZE}' for sia_pose_dino_simple. Use 'dino_b14' or 'dino_l14'.")
        model = get_sia_pose_dino_simple(
            size=dino_size,
            det_token_num=args.DET,
            num_frames=args.FRAMES,
            num_keypoints=17,
        )['sia']
        return model

    # ViCLIP-based models
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

    if args.MODEL == 'sia_pose':
        # Original model with pose decoder
        model = get_sia_pose(
            size=size,
            pretrain=pretrain,
            det_token_num=args.DET,
            num_frames=args.FRAMES,
            num_keypoints=17,
            pose_decoder_layers=args.POSE_LAYERS,
            enable_pose=True
        )['sia']
    elif args.MODEL == 'sia_pose_simple':
        # Simplified model without decoder
        model = get_sia_pose_simple(
            size=size,
            pretrain=pretrain,
            det_token_num=args.DET,
            num_frames=args.FRAMES,
            num_keypoints=17,
        )['sia']
    elif args.MODEL == 'sia_pose_heatmap':
        # ViTPose-style heatmap decoder: spatial features -> 2x Deconv -> Conv1x1 -> heatmaps
        model = get_sia_pose_heatmap(
            size=size,
            pretrain=pretrain,
            det_token_num=args.DET,
            num_frames=args.FRAMES,
            num_keypoints=17,
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
    if args.MODEL == 'sia_pose_heatmap':
        losses_list = ['keypoints_heatmap', 'boxes', 'human']
    else:
        losses_list = ['keypoints', 'boxes', 'human']

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.3, # Suppressing  false positives more
        losses=losses_list,
        num_keypoints=17  # COCO has 17 keypoints, needed for RLE loss sigma
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
            'pose_decoder', 'keypoint', 'pose_token', 'pose_positional',  # existing patterns
            'led_decoder', 'human_embed', 'bbox_embed',                    # LED/DINOv2 heads
            'temporal_positional_embedding', 'ln_post',                    # DINOv2-specific
            'heatmap_decoder',                                             # heatmap decoder
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


def build_optimizer(model, args):
    """Build optimizer with separate learning rates for backbone and heads."""
    if args.FREEZE_BACKBONE:
        # Backbone frozen — single group of trainable params
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.LR, weight_decay=args.WEIGHT_DECAY)
    else:
        # Backbone unfrozen — two param groups with different LRs
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if ('pose_decoder' in name or 'keypoint' in name or
                    'led_decoder' in name or 'human_head' in name or
                    'box_head' in name or 'det_token' in name or
                    'det_positional' in name or
                    'pose_token' in name or 'pose_positional' in name or
                    'human_embed' in name or 'bbox_embed' in name or
                    'temporal_positional_embedding' in name or 'ln_post' in name or
                    'heatmap_decoder' in name):
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.LR_BACKBONE},
            {'params': head_params, 'lr': args.LR},
        ], weight_decay=args.WEIGHT_DECAY)

    return optimizer


def train_one_epoch(model, criterion, weight_dict, dataloader, optimizer, epoch, rank, args, val_loader=None, postprocess=None, val_freq_batches=100, use_wandb=False):
    """Train for one epoch with periodic validation."""
    model.train()
    criterion.train()

    total_loss = 0
    num_batches = 0
    total_train_det = 0
    batch_stats = []
    num_total_batches = len(dataloader)
    log_freq = max(1, num_total_batches // 10)  # Log ~10 times per epoch when no tqdm
    wandb_log_freq = max(1, num_total_batches // 50)  # Log to wandb ~50 times per epoch

    # Use tqdm or simple iteration based on args
    if args.NO_TQDM:
        iterator = enumerate(dataloader)
    else:
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        # Track loss
        loss_value = losses.item()
        total_loss += loss_value
        num_batches += 1

        # Count detections
        if postprocess is not None:
            with torch.no_grad():
                imgsize = (args.HEIGHT, args.WIDTH)
                batch_results = postprocess(outputs, imgsize, human_conf=0.5, keypoint_conf=0.3)
                batch_det = sum(len(r['boxes']) for r in batch_results)
                total_train_det += batch_det

        # Log progress
        if rank == 0:
            loss_str = f"loss: {loss_value:.3f}"
            for k, v in loss_dict.items():
                if k in weight_dict:
                    loss_str += f" | {k}: {v.item():.3f}"

            if args.NO_TQDM:
                # Print progress periodically
                if (batch_idx + 1) % log_freq == 0 or batch_idx == 0:
                    print(f"  [{batch_idx + 1}/{num_total_batches}] {loss_str}", flush=True)
            else:
                # Update tqdm postfix - need to get pbar from iterator
                pass  # tqdm updates automatically with postfix set below

            # Log to wandb periodically
            if use_wandb and (batch_idx + 1) % wandb_log_freq == 0:
                global_step = epoch * num_total_batches + batch_idx
                wandb_metrics = {
                    'train/loss': loss_value,
                    'train/epoch': epoch + (batch_idx + 1) / num_total_batches,
                    'train/total_detections': total_train_det,
                }
                for k, v in loss_dict.items():
                    if k in weight_dict:
                        wandb_metrics[f'train/{k}'] = v.item()
                wandb.log(wandb_metrics, step=global_step)

        # Set postfix for tqdm (only works when using tqdm)
        if not args.NO_TQDM and rank == 0:
            try:
                iterator._iterable.set_postfix_str(loss_str[:80])
            except AttributeError:
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
                for k in ['AP', 'AP50', 'AP75', 'AR', 'mean_oks']:
                    if k in val_stats:
                        wandb_val[f'val/{k}'] = val_stats[k]
                wandb.log(wandb_val, step=global_step)
            model.train()  # Switch back to train mode

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, batch_stats, total_train_det


@torch.no_grad()
def validate(model, dataloader, postprocess, rank, args, criterion=None, weight_dict=None):
    """Validate the model using official COCO evaluation (pycocotools)."""
    model.eval()

    imgsize = (args.HEIGHT, args.WIDTH)
    num_keypoints = 17
    coco_results = []
    total_gt = 0
    total_det = 0
    total_val_loss = 0
    num_val_batches = 0
    num_images = 0
    total_oks = 0.0
    num_oks_matched = 0

    # Use tqdm or simple iteration based on args
    if args.NO_TQDM:
        iterator = dataloader
        if rank == 0:
            print("  Validating...", flush=True)
    else:
        iterator = tqdm(dataloader, desc="Validating", disable=(rank != 0), file=sys.stderr)

    for samples, targets in iterator:
        samples = samples.to(rank)

        outputs = model(samples)
        results = postprocess(outputs, imgsize, human_conf=0.5, keypoint_conf=0.3)

        for result, target in zip(results, targets):
            image_id = int(target['image_id'])
            num_images += 1
            total_gt += len(target['boxes'])

            pred_boxes = result['boxes'].cpu().numpy().astype(float)
            pred_kps = result['keypoints'].cpu().numpy().astype(float)
            pred_scores = result['scores'].cpu().numpy().astype(float)
            total_det += len(pred_boxes)

            # Rescale from resized coords to original image coords
            orig_h, orig_w = target['orig_size'].tolist()
            scale_x = orig_w / imgsize[1]
            scale_y = orig_h / imgsize[0]

            for pi in range(len(pred_boxes)):
                box = pred_boxes[pi]
                kps = pred_kps[pi]
                kps_flat = []
                for ki in range(num_keypoints):
                    kps_flat.extend([
                        float(kps[ki, 0]) * scale_x,
                        float(kps[ki, 1]) * scale_y,
                        2 if float(kps[ki, 2]) > 0.3 else 1,
                    ])
                coco_results.append({
                    'image_id': image_id,
                    'category_id': 1,
                    'keypoints': kps_flat,
                    'score': float(pred_scores[pi]),
                    'bbox': [
                        float(box[0]) * scale_x, float(box[1]) * scale_y,
                        float(box[2] - box[0]) * scale_x, float(box[3] - box[1]) * scale_y,
                    ],
                })

            # Compute per-instance OKS with greedy IoU matching
            gt_boxes_norm = target['boxes'].cpu().numpy().astype(float)  # [M, 4] cxcywh normalized
            gt_kps_norm = target['keypoints'].cpu().numpy().astype(float)  # [M, 17, 3] normalized

            if len(pred_boxes) > 0 and len(gt_boxes_norm) > 0:
                # Convert GT boxes: normalized cxcywh -> pixel xyxy
                gt_boxes_pixel = np.zeros_like(gt_boxes_norm)
                gt_boxes_pixel[:, 0] = (gt_boxes_norm[:, 0] - gt_boxes_norm[:, 2] / 2) * orig_w
                gt_boxes_pixel[:, 1] = (gt_boxes_norm[:, 1] - gt_boxes_norm[:, 3] / 2) * orig_h
                gt_boxes_pixel[:, 2] = (gt_boxes_norm[:, 0] + gt_boxes_norm[:, 2] / 2) * orig_w
                gt_boxes_pixel[:, 3] = (gt_boxes_norm[:, 1] + gt_boxes_norm[:, 3] / 2) * orig_h

                # Convert GT keypoints to pixel coords
                gt_kps_pixel = gt_kps_norm.copy()
                gt_kps_pixel[:, :, 0] *= orig_w
                gt_kps_pixel[:, :, 1] *= orig_h

                # GT bbox areas (w * h in pixels)
                gt_areas = (gt_boxes_pixel[:, 2] - gt_boxes_pixel[:, 0]) * \
                           (gt_boxes_pixel[:, 3] - gt_boxes_pixel[:, 1])

                # Pred boxes in pixel xyxy
                pred_boxes_pixel = np.zeros((len(pred_boxes), 4))
                pred_boxes_pixel[:, 0] = pred_boxes[:, 0] * scale_x
                pred_boxes_pixel[:, 1] = pred_boxes[:, 1] * scale_y
                pred_boxes_pixel[:, 2] = pred_boxes[:, 2] * scale_x
                pred_boxes_pixel[:, 3] = pred_boxes[:, 3] * scale_y

                # Greedy matching: preds sorted by score (already sorted from postprocess)
                matched_gt = set()
                for pi in range(len(pred_boxes_pixel)):
                    ious = box_iou_np(pred_boxes_pixel[pi], gt_boxes_pixel)
                    order = np.argsort(-ious)
                    for gi in order:
                        if gi not in matched_gt and ious[gi] > 0.0:
                            # Compute OKS for this (pred, gt) pair
                            pred_xy = np.stack([
                                pred_kps[pi, :, 0] * scale_x,
                                pred_kps[pi, :, 1] * scale_y,
                            ], axis=1)
                            oks = compute_oks(pred_xy, gt_kps_pixel[gi], gt_areas[gi], COCO_SIGMAS)
                            total_oks += oks
                            num_oks_matched += 1
                            matched_gt.add(gi)
                            break

        # Compute validation loss if criterion provided
        if criterion is not None and weight_dict is not None:
            for t in targets:
                t['boxes'] = t['boxes'].to(rank)
                t['keypoints'] = t['keypoints'].to(rank)
                t['labels'] = torch.zeros(len(t['boxes']), 1).to(rank)

            loss_dict = criterion(outputs, targets, num_classes=1)
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            total_val_loss += losses.item()
            num_val_batches += 1

    val_loss = total_val_loss / max(num_val_batches, 1) if num_val_batches > 0 else 0.0
    mean_oks = total_oks / max(num_oks_matched, 1)

    # Run official COCO evaluation (only on rank 0 to avoid duplicate output)
    coco_stats = {}
    if rank == 0:
        print(f"Validation: {num_images} images, {total_gt} GT persons, {total_det} detections, "
              f"val_loss: {val_loss:.4f}, mean_oks: {mean_oks:.4f} ({num_oks_matched} matched)", flush=True)
        coco_stats = run_coco_eval(args.VAL_ANN, coco_results, sigmas=COCO_SIGMAS)

    return {
        'total_gt': total_gt,
        'total_det': total_det,
        'val_loss': val_loss,
        'mean_oks': mean_oks,
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

    # Resume from checkpoint
    if args.RESUME:
        if is_main_process():
            print(f"Resuming from {args.RESUME}", flush=True)
        state_dict = torch.load(args.RESUME, map_location='cpu', weights_only=False)

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

    # Build optimizer
    optimizer = build_optimizer(model, args)

    # Build scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.EPOCH, eta_min=1e-6)

    # Build datasets
    if is_main_process():
        print("Loading datasets...", flush=True)

    transforms = v2.Compose([
        v2.Resize((args.HEIGHT, args.WIDTH)),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = COCOPose(
        root=os.path.join(args.COCO_ROOT, 'train2017'),
        annFile=args.TRAIN_ANN,
        transforms=transforms,
        frames=args.FRAMES,
        min_keypoints=args.MIN_KP,
        augment=True,
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
        val_dataset = COCOPoseVal(
            root=os.path.join(args.COCO_ROOT, 'val2017'),
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
        print(f"\nStarting training for {args.EPOCH} epochs...", flush=True)
        print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset) if not args.NO_VAL else 0} images", flush=True)
        print(f"Batch size: {args.BS} x {world_size} GPUs = {args.BS * world_size} effective", flush=True)

    best_loss = float('inf')
    train_stats = []

    for epoch in range(args.EPOCH):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        avg_loss, batch_stats, total_train_det = train_one_epoch(
            model, criterion, weight_dict, train_loader, optimizer, epoch, rank, args,
            val_loader=val_loader if not args.NO_VAL else None,
            postprocess=postprocess,
            val_freq_batches=args.VAL_BATCH_FREQ,
            use_wandb=use_wandb
        )

        scheduler.step()

        if is_main_process():
            print(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}, lr = {scheduler.get_last_lr()[0]:.2e}, train_det: {total_train_det}", flush=True)
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
                'epoch/train_detections': total_train_det,
                'epoch': epoch,
            }
            if val_stats is not None:
                epoch_metrics['epoch/val_loss'] = val_stats['val_loss']
                for k in ['AP', 'AP50', 'AP75', 'AR', 'mean_oks']:
                    if k in val_stats:
                        epoch_metrics[f'epoch/{k}'] = val_stats[k]
            wandb.log(epoch_metrics)

        # Save checkpoint
        if args.SAVE and is_main_process():
            # Get state dict (handle DDP vs non-DDP model)
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            if (epoch + 1) % args.SAVE_FREQ == 0:
                ckpt_path = os.path.join(exp_dir, f'{args.MODEL}_{args.SIZE}_epoch{epoch}.pt')
                torch.save(state_dict, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}", flush=True)

            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(exp_dir, f'{args.MODEL}_{args.SIZE}_best.pt')
                torch.save(state_dict, best_path)

    # Save final
    if args.SAVE and is_main_process():
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        final_path = os.path.join(exp_dir, f'{args.MODEL}_{args.SIZE}_final.pt')
        torch.save(state_dict, final_path)
        print(f"Saved final model: {final_path}", flush=True)

        # Save training stats
        stats_path = os.path.join(exp_dir, 'train_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(train_stats, f, indent=2)

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
    print(f"COCO root: {args.COCO_ROOT}")
    
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
