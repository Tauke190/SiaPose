"""
Standalone evaluation script for SIA Pose Estimation models.

Supports:
  - All model variants: sia_pose, sia_pose_simple, sia_pose_decoder_led
  - Multiple datasets: COCO (default), extensible to others
  - Official COCO evaluation via pycocotools (AP, AP50, AP75, AR)
  - Custom OKS-based evaluation with per-keypoint breakdown
  - Saving predictions in COCO results format for external tools

Usage:
  python val_pose.py \
      --checkpoint weights/model.pt \
      --model sia_pose_simple \
      --dataset coco \
      --data_root /path/to/coco \
      --ann_file /path/to/person_keypoints_val2017.json
"""
import os
import sys
import json
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from sia import (
    get_sia_pose, get_sia_pose_simple, get_sia_pose_decoder_led,
    PostProcessPose, COCO_KEYPOINT_NAMES,
)
from datasets import COCOPoseVal

# COCO keypoint sigmas for OKS computation (17 keypoints)
COCO_SIGMAS = np.array([
    0.026,                  # nose
    0.025, 0.025,           # left_eye, right_eye
    0.035, 0.035,           # left_ear, right_ear
    0.079, 0.079,           # left_shoulder, right_shoulder
    0.072, 0.072,           # left_elbow, right_elbow
    0.062, 0.062,           # left_wrist, right_wrist
    0.107, 0.107,           # left_hip, right_hip
    0.087, 0.087,           # left_knee, right_knee
    0.089, 0.089,           # left_ankle, right_ankle
])


# ---------------------------------------------------------------------------
# Dataset registry — add new datasets here
# ---------------------------------------------------------------------------

def get_dataset(args, transforms):
    """Build the evaluation dataset based on --dataset flag.

    Returns:
        dataset: a torch Dataset that yields (clip, target) pairs.
                 Each target dict must contain at least:
                   - 'image_id': int
                   - 'boxes': Tensor [N, 4] normalised cxcywh
                   - 'keypoints': Tensor [N, 17, 3] normalised (x, y, vis)
        ann_file: path to the annotation JSON (needed for official COCO eval)
        num_keypoints: int
        keypoint_names: list[str]
        sigmas: np.array of keypoint sigmas
    """
    name = args.dataset.lower()

    if name == 'coco':
        ann_file = args.ann_file
        if ann_file is None:
            ann_file = os.path.join(args.data_root, 'annotations',
                                    'person_keypoints_val2017.json')
        img_dir = args.img_dir
        if img_dir is None:
            img_dir = os.path.join(args.data_root, 'val2017')

        dataset = COCOPoseVal(
            root=img_dir,
            annFile=ann_file,
            transforms=transforms,
            frames=args.num_frames,
            min_keypoints=args.min_kp,
        )
        return dataset, ann_file, 17, COCO_KEYPOINT_NAMES, COCO_SIGMAS

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: coco. "
            "To add a new dataset, extend get_dataset() in val_pose.py."
        )


# ---------------------------------------------------------------------------
# Model builder (mirrors train_pose.py / inference_image.py)
# ---------------------------------------------------------------------------

def build_model(args):
    """Instantiate the model (without pretrained backbone weights)."""
    size = 'b' if args.size == 'b16' else 'l'

    if args.model == 'sia_pose':
        model = get_sia_pose(
            size=size, pretrain=None,
            det_token_num=args.det_tokens,
            num_frames=args.num_frames,
            num_keypoints=17,
            pose_decoder_layers=args.pose_layers,
            enable_pose=True,
        )['sia']
    elif args.model == 'sia_pose_simple':
        model = get_sia_pose_simple(
            size=size, pretrain=None,
            det_token_num=args.det_tokens,
            num_frames=args.num_frames,
            num_keypoints=17,
        )['sia']
    elif args.model == 'sia_pose_decoder_led':
        model = get_sia_pose_decoder_led(
            size=size, pretrain=None,
            det_token_num=args.det_tokens,
            num_frames=args.num_frames,
            num_keypoints=17,
            decoder_layers=args.pose_layers,
        )['sia']
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model


# ---------------------------------------------------------------------------
# OKS helpers (custom, for per-keypoint breakdown)
# ---------------------------------------------------------------------------

def compute_oks(pred_kp, gt_kp, gt_area, sigmas):
    """Compute OKS between a single prediction and ground truth."""
    vis_mask = gt_kp[:, 2] > 0
    if vis_mask.sum() == 0:
        return 0.0

    dx = pred_kp[:, 0] - gt_kp[:, 0]
    dy = pred_kp[:, 1] - gt_kp[:, 1]
    d_sq = dx ** 2 + dy ** 2

    vars_ = (sigmas * 2) ** 2
    exp_terms = np.exp(-d_sq / (2 * gt_area * vars_ + 1e-8))
    return float(np.sum(exp_terms * vis_mask) / np.sum(vis_mask))


def compute_per_keypoint_accuracy(pred_kp, gt_kp, gt_area, sigmas,
                                  oks_thresh=0.5):
    """Return a boolean array [num_keypoints] indicating correct keypoints."""
    vars_ = (sigmas * 2) ** 2
    dx = pred_kp[:, 0] - gt_kp[:, 0]
    dy = pred_kp[:, 1] - gt_kp[:, 1]
    d_sq = dx ** 2 + dy ** 2
    per_kp_oks = np.exp(-d_sq / (2 * gt_area * vars_ + 1e-8))
    return per_kp_oks >= oks_thresh


def compute_iou(box1, box2):
    """IoU between two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-8)


def compute_ap_oks(all_preds, all_gts, sigmas, oks_thresh=0.5,
                   iou_thresh=0.5):
    """Compute AP at a given OKS threshold (custom implementation)."""
    all_detections = []
    total_gt = 0

    for preds, gts in zip(all_preds, all_gts):
        num_gt = len(gts['boxes'])
        total_gt += num_gt
        if num_gt == 0:
            for s in preds['scores']:
                all_detections.append((s, False))
            continue
        if len(preds['boxes']) == 0:
            continue

        gt_matched = [False] * num_gt
        order = np.argsort(-preds['scores'])

        for pi in order:
            best_oks, best_gi = 0.0, -1
            for gi in range(num_gt):
                if gt_matched[gi]:
                    continue
                iou = compute_iou(preds['boxes'][pi], gts['boxes'][gi])
                if iou < iou_thresh:
                    continue
                oks = compute_oks(preds['keypoints'][pi], gts['keypoints'][gi],
                                  gts['areas'][gi], sigmas)
                if oks > best_oks:
                    best_oks = oks
                    best_gi = gi
            is_tp = best_oks >= oks_thresh and best_gi >= 0
            if is_tp:
                gt_matched[best_gi] = True
            all_detections.append((preds['scores'][pi], is_tp))

    if total_gt == 0 or len(all_detections) == 0:
        return 0.0

    all_detections.sort(key=lambda x: -x[0])
    tp, fp = 0, 0
    precs, recs = [], []
    for _, is_tp in all_detections:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precs.append(tp / (tp + fp))
        recs.append(tp / total_gt)

    # 101-pt interpolation (COCO style)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = max((pr for pr, rc in zip(precs, recs) if rc >= t), default=0.0)
        ap += p / 101
    return ap


# ---------------------------------------------------------------------------
# Official COCO evaluation via pycocotools
# ---------------------------------------------------------------------------

def run_coco_eval(ann_file, coco_results, output_dir=None):
    """Run the official COCO keypoint evaluation.

    Args:
        ann_file: path to COCO ground-truth annotation JSON
        coco_results: list of dicts in COCO keypoint result format
        output_dir: if provided, save the results JSON here

    Returns:
        stats dict with AP, AP50, AP75, AR, etc.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if len(coco_results) == 0:
        print("No predictions to evaluate.")
        return {}

    # Save predictions to temp file
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        res_path = os.path.join(output_dir, 'coco_results.json')
    else:
        res_path = '/tmp/sia_pose_coco_results.json'

    with open(res_path, 'w') as f:
        json.dump(coco_results, f)

    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(res_path)

    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    names = ['AP', 'AP50', 'AP75', 'AP_M', 'AP_L', 'AR', 'AR50', 'AR75',
             'AR_M', 'AR_L']
    stats = {n: float(v) for n, v in zip(names, coco_eval.stats)}
    return stats


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def collate_fn(batch):
    clips, targets = zip(*batch)
    return torch.stack(clips), list(targets)


@torch.no_grad()
def evaluate(model, dataloader, postprocess, args, sigmas, keypoint_names):
    """Run evaluation on the full dataset.

    Returns:
        metrics: dict of scalar metrics
        coco_results: list of COCO-format result dicts
        all_preds / all_gts: raw per-image predictions and ground truths
    """
    model.eval()
    device = next(model.parameters()).device
    imgsize = (args.height, args.width)

    all_preds, all_gts = [], []
    coco_results = []
    num_keypoints = len(sigmas)

    # Per-keypoint trackers
    kp_correct = np.zeros(num_keypoints)
    kp_total = np.zeros(num_keypoints)

    total_inference_time = 0.0
    num_images = 0

    for samples, targets in tqdm(dataloader, desc="Evaluating", file=sys.stderr):
        samples = samples.to(device)
        bs = samples.shape[0]

        t0 = time.time()
        outputs = model(samples)
        total_inference_time += time.time() - t0

        results = postprocess(outputs, imgsize,
                              human_conf=args.conf_thresh,
                              keypoint_conf=args.kp_conf_thresh)

        for i in range(bs):
            result = results[i]
            target = targets[i]
            image_id = int(target['image_id'])
            num_images += 1

            # --- predictions in pixel coords ---
            pred_boxes = result['boxes'].cpu().numpy().astype(float)
            pred_kps = result['keypoints'].cpu().numpy().astype(float)
            pred_scores = result['scores'].cpu().numpy().astype(float)

            all_preds.append({
                'boxes': pred_boxes,
                'keypoints': pred_kps,
                'scores': pred_scores,
            })

            # --- ground truth: normalised → pixel ---
            gt_boxes_norm = target['boxes'].cpu().numpy()
            gt_kps_norm = target['keypoints'].cpu().numpy()

            gt_boxes_px, gt_areas = [], []
            for box in gt_boxes_norm:
                cx, cy, w, h = box
                x1 = (cx - w / 2) * imgsize[1]
                y1 = (cy - h / 2) * imgsize[0]
                x2 = (cx + w / 2) * imgsize[1]
                y2 = (cy + h / 2) * imgsize[0]
                gt_boxes_px.append([x1, y1, x2, y2])
                gt_areas.append(w * imgsize[1] * h * imgsize[0])

            gt_boxes_px = np.array(gt_boxes_px) if gt_boxes_px else np.zeros((0, 4))
            gt_areas = np.array(gt_areas) if gt_areas else np.zeros(0)

            gt_kps_px = gt_kps_norm.copy()
            if len(gt_kps_px) > 0:
                gt_kps_px[:, :, 0] *= imgsize[1]
                gt_kps_px[:, :, 1] *= imgsize[0]

            all_gts.append({
                'boxes': gt_boxes_px,
                'keypoints': gt_kps_px,
                'areas': gt_areas,
            })

            # --- per-keypoint accuracy (matched by IoU) ---
            if len(pred_boxes) > 0 and len(gt_boxes_px) > 0:
                gt_matched = [False] * len(gt_boxes_px)
                order = np.argsort(-pred_scores)
                for pi in order:
                    best_gi = -1
                    best_iou = 0.5  # threshold
                    for gi in range(len(gt_boxes_px)):
                        if gt_matched[gi]:
                            continue
                        iou = compute_iou(pred_boxes[pi], gt_boxes_px[gi])
                        if iou > best_iou:
                            best_iou = iou
                            best_gi = gi
                    if best_gi >= 0:
                        gt_matched[best_gi] = True
                        vis_mask = gt_kps_px[best_gi][:, 2] > 0
                        correct = compute_per_keypoint_accuracy(
                            pred_kps[pi], gt_kps_px[best_gi],
                            gt_areas[best_gi], sigmas)
                        kp_correct += correct * vis_mask
                        kp_total += vis_mask

            # --- build COCO results (for official eval) ---
            for pi in range(len(pred_boxes)):
                box = pred_boxes[pi]
                kps = pred_kps[pi]
                # COCO expects keypoints as [x1,y1,v1,...,x17,y17,v17]
                kps_flat = []
                for ki in range(num_keypoints):
                    kps_flat.extend([
                        float(kps[ki, 0]),
                        float(kps[ki, 1]),
                        # COCO visibility: 0=not labeled, 1=labeled not visible, 2=labeled visible
                        2 if float(kps[ki, 2]) > args.kp_conf_thresh else 1,
                    ])
                coco_results.append({
                    'image_id': image_id,
                    'category_id': 1,  # person
                    'keypoints': kps_flat,
                    'score': float(pred_scores[pi]),
                    'bbox': [
                        float(box[0]), float(box[1]),
                        float(box[2] - box[0]), float(box[3] - box[1]),
                    ],  # xywh
                })

    # --- aggregate metrics ---
    total_gt = sum(len(g['boxes']) for g in all_gts)
    total_det = sum(len(p['boxes']) for p in all_preds)
    fps = num_images / total_inference_time if total_inference_time > 0 else 0

    # Custom AP at different OKS thresholds
    ap50 = compute_ap_oks(all_preds, all_gts, sigmas, oks_thresh=0.5)
    ap75 = compute_ap_oks(all_preds, all_gts, sigmas, oks_thresh=0.75)

    # AP averaged over thresholds 0.50:0.05:0.95 (COCO style)
    oks_thresholds = np.arange(0.50, 1.00, 0.05)
    ap_per_thresh = [compute_ap_oks(all_preds, all_gts, sigmas, t)
                     for t in oks_thresholds]
    ap_mean = float(np.mean(ap_per_thresh))

    # Per-keypoint accuracy
    kp_acc = {}
    for ki in range(num_keypoints):
        acc = kp_correct[ki] / max(kp_total[ki], 1)
        kp_acc[keypoint_names[ki]] = float(acc)

    metrics = {
        'num_images': num_images,
        'total_gt': total_gt,
        'total_detections': total_det,
        'fps': round(fps, 1),
        'custom_AP': round(ap_mean, 4),
        'custom_AP50': round(ap50, 4),
        'custom_AP75': round(ap75, 4),
        'per_keypoint_accuracy': kp_acc,
    }

    return metrics, coco_results, all_preds, all_gts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate SIA Pose Estimation models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to model checkpoint (.pt)')
    p.add_argument('--model', type=str, default='sia_pose_simple',
                   choices=['sia_pose', 'sia_pose_simple', 'sia_pose_decoder_led'],
                   help='Model architecture')
    p.add_argument('--size', type=str, default='b16', choices=['b16', 'l14'],
                   help='Model size')
    p.add_argument('--det_tokens', type=int, default=100,
                   help='Number of detection tokens')
    p.add_argument('--pose_layers', type=int, default=2,
                   help='Number of pose decoder layers')
    p.add_argument('--num_frames', type=int, default=1,
                   help='Number of input frames (image duplicated)')

    # Dataset
    p.add_argument('--dataset', type=str, default='coco',
                   help='Dataset name (coco)')
    p.add_argument('--data_root', type=str, default=None,
                   help='Dataset root directory')
    p.add_argument('--img_dir', type=str, default=None,
                   help='Image directory (overrides data_root/val2017)')
    p.add_argument('--ann_file', type=str, default=None,
                   help='Annotation JSON path (overrides data_root default)')
    p.add_argument('--min_kp', type=int, default=1,
                   help='Minimum visible keypoints per person to include')

    # Input
    p.add_argument('--width', type=int, default=320, help='Input width')
    p.add_argument('--height', type=int, default=240, help='Input height')

    # Inference
    p.add_argument('--batch_size', type=int, default=32, help='Batch size')
    p.add_argument('--workers', type=int, default=8,
                   help='DataLoader workers')
    p.add_argument('--conf_thresh', type=float, default=0.5,
                   help='Human detection confidence threshold')
    p.add_argument('--kp_conf_thresh', type=float, default=0.3,
                   help='Keypoint visibility confidence threshold')
    p.add_argument('--device', type=str, default=None,
                   help='Device (cuda:0, cpu). Default: auto-detect')

    # Output
    p.add_argument('--output_dir', type=str, default=None,
                   help='Directory to save results JSON and metrics')
    p.add_argument('--no_coco_eval', action='store_true',
                   help='Skip official COCO evaluation (pycocotools)')

    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*60}")
    print(f"SIA Pose Evaluation")
    print(f"{'='*60}")
    print(f"Model       : {args.model} ({args.size})")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Dataset     : {args.dataset}")
    print(f"Device      : {device}")
    print(f"Input size  : {args.width}x{args.height}")
    print(f"Conf thresh : human={args.conf_thresh}, kp={args.kp_conf_thresh}")
    print(f"{'='*60}")

    # Build model & load weights
    print("\n[1/4] Building model...")
    model = build_model(args)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"      Loaded weights from {args.checkpoint}")

    # Build dataset
    print("\n[2/4] Loading dataset...")
    transforms = v2.Compose([
        v2.Resize((args.height, args.width)),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset, ann_file, num_kp, kp_names, sigmas = get_dataset(args, transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
    )
    print(f"      {len(dataset)} images, {num_kp} keypoints")

    # Run evaluation
    print("\n[3/4] Running evaluation...")
    postprocess = PostProcessPose()
    metrics, coco_results, all_preds, all_gts = evaluate(
        model, dataloader, postprocess, args, sigmas, kp_names,
    )

    # Print custom metrics
    print(f"\n{'='*60}")
    print("Custom Metrics (computed internally)")
    print(f"{'='*60}")
    print(f"  Images evaluated  : {metrics['num_images']}")
    print(f"  Total GT persons  : {metrics['total_gt']}")
    print(f"  Total detections  : {metrics['total_detections']}")
    print(f"  Inference speed   : {metrics['fps']:.1f} img/s")
    print(f"  AP  (OKS 0.50:0.95) : {metrics['custom_AP']:.4f}")
    print(f"  AP50 (OKS 0.50)     : {metrics['custom_AP50']:.4f}")
    print(f"  AP75 (OKS 0.75)     : {metrics['custom_AP75']:.4f}")

    print(f"\n  Per-Keypoint Accuracy (OKS >= 0.5):")
    print(f"  {'Keypoint':<20} {'Accuracy':>8}")
    print(f"  {'-'*30}")
    for kp_name, acc in metrics['per_keypoint_accuracy'].items():
        print(f"  {kp_name:<20} {acc:>8.4f}")

    # Official COCO eval
    coco_stats = {}
    if not args.no_coco_eval and ann_file is not None:
        print(f"\n{'='*60}")
        print("Official COCO Evaluation (pycocotools)")
        print(f"{'='*60}")
        coco_stats = run_coco_eval(ann_file, coco_results, args.output_dir)
        metrics['coco_eval'] = coco_stats

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[4/4] Metrics saved to {metrics_path}")

        results_path = os.path.join(args.output_dir, 'coco_results.json')
        with open(results_path, 'w') as f:
            json.dump(coco_results, f)
        print(f"      COCO results saved to {results_path}")
    else:
        print(f"\n[4/4] Tip: use --output_dir to save results to disk")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
