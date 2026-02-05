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

from sia import get_sia_pose, get_sia_pose_simple, get_sia_pose_decoder_led, HungarianMatcher, SetCriterion, PostProcessPose
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


def compute_oks(pred_kp, gt_kp, gt_area):
    """Compute Object Keypoint Similarity between predicted and ground truth keypoints.

    Args:
        pred_kp: Predicted keypoints [num_keypoints, 3] (x, y, vis) - pixel coordinates
        gt_kp: Ground truth keypoints [num_keypoints, 3] (x, y, vis) - pixel coordinates
        gt_area: Ground truth bounding box area (in pixels)

    Returns:
        OKS score in [0, 1]
    """
    # Get visibility mask from ground truth (vis > 0 means labeled)
    vis_mask = gt_kp[:, 2] > 0

    if vis_mask.sum() == 0:
        return 0.0

    # Compute squared distances for visible keypoints
    dx = pred_kp[:, 0] - gt_kp[:, 0]
    dy = pred_kp[:, 1] - gt_kp[:, 1]
    d_squared = dx ** 2 + dy ** 2

    # Scale factor: s^2 = area, so we use area directly
    # OKS uses k^2 * 2 * s^2 as the denominator variance
    s_squared = gt_area
    vars = (COCO_SIGMAS * 2) ** 2

    # Compute OKS: sum of exp(-d^2 / (2 * s^2 * k^2)) for visible keypoints
    exp_terms = np.exp(-d_squared / (2 * s_squared * vars + 1e-8))

    # Only count visible keypoints
    oks = np.sum(exp_terms * vis_mask) / np.sum(vis_mask)

    return oks


def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-8)


def compute_ap_oks(all_predictions, all_targets, oks_threshold=0.5, iou_threshold=0.5):
    """Compute Average Precision at a given OKS threshold.

    Args:
        all_predictions: List of dicts with 'boxes', 'keypoints', 'scores' (pixel coords)
        all_targets: List of dicts with 'boxes', 'keypoints', 'area' (pixel coords)
        oks_threshold: OKS threshold for true positive
        iou_threshold: IoU threshold for box matching

    Returns:
        AP score
    """
    # Collect all predictions with their scores and match info
    all_detections = []  # (score, is_tp, oks)
    total_gt = 0

    for preds, targets in zip(all_predictions, all_targets):
        pred_boxes = preds['boxes']
        pred_kps = preds['keypoints']
        pred_scores = preds['scores']

        gt_boxes = targets['boxes']
        gt_kps = targets['keypoints']
        gt_areas = targets['areas']

        num_gt = len(gt_boxes)
        total_gt += num_gt

        if num_gt == 0:
            # All predictions are false positives
            for score in pred_scores:
                all_detections.append((score, False, 0.0))
            continue

        if len(pred_boxes) == 0:
            continue

        # Track which GT have been matched
        gt_matched = [False] * num_gt

        # Sort predictions by score (descending)
        sorted_indices = np.argsort(-pred_scores)

        for pred_idx in sorted_indices:
            pred_box = pred_boxes[pred_idx]
            pred_kp = pred_kps[pred_idx]
            pred_score = pred_scores[pred_idx]

            best_oks = 0.0
            best_gt_idx = -1

            # Find best matching GT (by IoU first, then compute OKS)
            for gt_idx in range(num_gt):
                if gt_matched[gt_idx]:
                    continue

                iou = compute_iou(pred_box, gt_boxes[gt_idx])
                if iou < iou_threshold:
                    continue

                # Compute OKS
                oks = compute_oks(pred_kp, gt_kps[gt_idx], gt_areas[gt_idx])

                if oks > best_oks:
                    best_oks = oks
                    best_gt_idx = gt_idx

            # Check if it's a true positive
            is_tp = best_oks >= oks_threshold and best_gt_idx >= 0
            if is_tp:
                gt_matched[best_gt_idx] = True

            all_detections.append((pred_score, is_tp, best_oks))

    if total_gt == 0 or len(all_detections) == 0:
        return 0.0

    # Sort by score descending
    all_detections.sort(key=lambda x: -x[0])

    # Compute precision-recall curve
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for score, is_tp, oks in all_detections:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt

        precisions.append(precision)
        recalls.append(recall)

    # Compute AP using 101-point interpolation (COCO style)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        # Find precision at recall >= t
        p = 0.0
        for prec, rec in zip(precisions, recalls):
            if rec >= t:
                p = max(p, prec)
        ap += p / 101

    return ap


def parse_args():
    parser = argparse.ArgumentParser(description="Train SIA for Pose Estimation on COCO")

    # Model
    parser.add_argument("-MODEL", type=str, default='pose', choices=['sia_pose', 'sia_pose_simple', 'sia_pose_decoder_led'],
                        help="Model type: 'sia_pose' (det tokens in encoder), 'sia_pose_simple' (no decoder), 'sia_pose_decoder_led' (DETR-style LED)")
    parser.add_argument("-SIZE", type=str, default='b16', choices=['b16', 'l14'],
                        help="Model size: b16 (ViT-B/16) or l14 (ViT-L/14)")
    parser.add_argument("-FRAMES", type=int, default=9,
                        help="Number of input frames (image duplicated)")
    parser.add_argument("-DET", type=int, default=20,
                        help="Number of detection tokens")
    parser.add_argument("-POSE_LAYERS", type=int, default=2,
                        help="Number of pose decoder layers")

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
    parser.add_argument("-W_KP", type=float, default=5.0,
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
    if args.SIZE == 'b16':
        pretrain = "weights/ViCLIP-B_InternVid-FLT-10M.pth"
        size = 'b'
    else:
        pretrain = "weights/ViCLIP-L_InternVid-FLT-10M.pth"
        size = 'l'

    # Check if pretrain exists, otherwise use None
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
    elif args.MODEL == 'sia_pose_decoder_led':
        # Late Encoder-Decoder (LED) model: no det tokens in encoder, decoder cross-attends to spatial features
        model = get_sia_pose_decoder_led(
            size=size,
            pretrain=pretrain,
            det_token_num=args.DET,
            num_frames=args.FRAMES,
            num_keypoints=17,
            decoder_layers=args.POSE_LAYERS,
        )['sia']
    else:
        raise ValueError(f"Unknown model type: {args.MODEL}")

    return model


def build_criterion(args, rank):
    """Build matcher and criterion for training."""
    matcher = HungarianMatcher(
        cost_class=0,  # No action classes
        cost_bbox=args.W_BBOX,
        cost_giou=args.W_GIOU,
        cost_human=args.W_HUMAN,
        cost_keypoint=args.COST_KP
    )

    # Include bbox and human losses so all parameters receive gradients (required for DDP)
    weight_dict = {
        'loss_keypoints': args.W_KP,
        'loss_keypoint_vis': args.W_KP_VIS,
        'loss_bbox': args.W_BBOX,
        'loss_giou': args.W_GIOU,
        'loss_human': args.W_HUMAN,
    }

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=['keypoints', 'boxes', 'human'],  # All losses for DDP compatibility
        num_keypoints=17  # COCO has 17 keypoints, needed for RLE loss sigma
    )
    criterion.to(rank)

    return criterion, weight_dict


def freeze_encoder(model):
    """Freeze encoder/backbone parameters, keeping only pose decoder trainable."""
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        # Keep pose decoder and keypoint-related parameters trainable
        if 'pose_decoder' in name or 'keypoint' in name:
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
    """Build optimizer with different learning rates for backbone and decoder."""
    # Only include parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=args.LR, weight_decay=args.WEIGHT_DECAY)

    return optimizer


def train_one_epoch(model, criterion, weight_dict, dataloader, optimizer, epoch, rank, args, val_loader=None, postprocess=None, val_freq_batches=100, use_wandb=False):
    """Train for one epoch with periodic validation."""
    model.train()
    criterion.train()

    total_loss = 0
    num_batches = 0
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
        if val_loader is not None and (batch_idx + 1) % val_freq_batches == 0:
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
                wandb.log({
                    'val/loss': val_stats['val_loss'],
                    'val/ap_oks_50': val_stats['ap_oks_50'],
                    'val/total_detections': val_stats['total_det'],
                    'val/total_gt': val_stats['total_gt'],
                }, step=global_step)
            model.train()  # Switch back to train mode

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, batch_stats


@torch.no_grad()
def validate(model, dataloader, postprocess, rank, args, criterion=None, weight_dict=None):
    """Validate the model with AP@OKS=0.5 metric."""
    model.eval()

    all_results = []
    all_predictions = []
    all_targets = []
    imgsize = (args.HEIGHT, args.WIDTH)
    total_val_loss = 0
    num_val_batches = 0

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

        for i, (result, target) in enumerate(zip(results, targets)):
            all_results.append({
                'image_id': target['image_id'],
                'num_detections': len(result['boxes']),
                'num_gt': len(target['boxes']),
            })

            # Collect predictions for AP computation (pixel coordinates)
            pred_boxes = result['boxes'].cpu().numpy()  # [N, 4] xyxy
            pred_kps = result['keypoints'].cpu().numpy()  # [N, 17, 3]
            pred_scores = result['scores'].cpu().numpy()  # [N]

            all_predictions.append({
                'boxes': pred_boxes,
                'keypoints': pred_kps,
                'scores': pred_scores,
            })

            # Convert GT to pixel coordinates (from normalized cxcywh)
            gt_boxes_norm = target['boxes'].cpu().numpy()  # [M, 4] cxcywh normalized
            gt_kps_norm = target['keypoints'].cpu().numpy()  # [M, 17, 3] normalized

            # Convert GT boxes: cxcywh normalized -> xyxy pixel
            gt_boxes_pixel = []
            gt_areas = []
            for box in gt_boxes_norm:
                cx, cy, w, h = box
                x1 = (cx - w / 2) * imgsize[1]
                y1 = (cy - h / 2) * imgsize[0]
                x2 = (cx + w / 2) * imgsize[1]
                y2 = (cy + h / 2) * imgsize[0]
                gt_boxes_pixel.append([x1, y1, x2, y2])
                # Area in pixels for OKS
                gt_areas.append(w * imgsize[1] * h * imgsize[0])

            gt_boxes_pixel = np.array(gt_boxes_pixel) if gt_boxes_pixel else np.zeros((0, 4))
            gt_areas = np.array(gt_areas) if gt_areas else np.zeros(0)

            # Convert GT keypoints to pixel coordinates
            gt_kps_pixel = gt_kps_norm.copy()
            if len(gt_kps_pixel) > 0:
                gt_kps_pixel[:, :, 0] *= imgsize[1]  # x * width
                gt_kps_pixel[:, :, 1] *= imgsize[0]  # y * height

            all_targets.append({
                'boxes': gt_boxes_pixel,
                'keypoints': gt_kps_pixel,
                'areas': gt_areas,
            })

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

    # Simple metrics: detection recall
    total_gt = sum(r['num_gt'] for r in all_results)
    total_det = sum(r['num_detections'] for r in all_results)

    val_loss = total_val_loss / max(num_val_batches, 1) if num_val_batches > 0 else 0.0

    # Compute AP@OKS=0.5
    ap_oks_50 = compute_ap_oks(all_predictions, all_targets, oks_threshold=0.5, iou_threshold=0.5)

    if rank == 0:
        loss_str = f"Validation: {len(all_results)} images, {total_gt} GT persons, {total_det} detections"
        if num_val_batches > 0:
            loss_str += f", AP@OKS=0.5: {ap_oks_50:.4f}"
            loss_str += f", val_loss: {val_loss:.4f}"
        print(loss_str, flush=True)

    return {'total_gt': total_gt, 'total_det': total_det, 'val_loss': val_loss, 'ap_oks_50': ap_oks_50}


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
        state_dict = torch.load(args.RESUME, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    # Freeze encoder parameters
    if is_main_process():
        print("Freezing encoder parameters...", flush=True)
    freeze_encoder(model)

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
        min_keypoints=args.MIN_KP
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
    exp_dir = os.path.join('weights', args.EXP)
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
        avg_loss, batch_stats = train_one_epoch(
            model, criterion, weight_dict, train_loader, optimizer, epoch, rank, args,
            val_loader=val_loader if not args.NO_VAL else None,
            postprocess=postprocess if not args.NO_VAL else None,
            val_freq_batches=100,
            use_wandb=use_wandb
        )

        scheduler.step()

        if is_main_process():
            print(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}, lr = {scheduler.get_last_lr()[0]:.2e}", flush=True)
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
                epoch_metrics.update({
                    'epoch/val_loss': val_stats['val_loss'],
                    'epoch/ap_oks_50': val_stats['ap_oks_50'],
                })
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
