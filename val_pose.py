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
    get_sia_pose, get_sia_pose_simple,
    PostProcessPose, COCO_KEYPOINT_NAMES,
)
from datasets import COCOPoseVal, PoseTrackPoseVal

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

# PoseTrack 2017 keypoint names (15 keypoints)
POSETRACK_KEYPOINT_NAMES = [
    'nose', 'head_bottom', 'head_top',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]

# PoseTrack 2017 sigmas for OKS computation (15 keypoints)
# Shared keypoints use COCO sigmas; head_bottom/head_top use ear-like sigma
POSETRACK_SIGMAS = np.array([
    0.026,                  # nose
    0.035,                  # head_bottom (approx, similar to ear)
    0.035,                  # head_top (approx, similar to ear)
    0.079, 0.079,           # left_shoulder, right_shoulder
    0.072, 0.072,           # left_elbow, right_elbow
    0.062, 0.062,           # left_wrist, right_wrist
    0.107, 0.107,           # left_hip, right_hip
    0.087, 0.087,           # left_knee, right_knee
    0.089, 0.089,           # left_ankle, right_ankle
])


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


def map_coco17_to_posetrack15(pred_kps_17):
    """Map model's 17 COCO keypoints to PoseTrack's 15 keypoints.

    Direct mapping for 13 shared keypoints (nose + body).
    Approximates head_bottom (midpoint of shoulders) and head_top
    (reflection of head_bottom through nose).

    Args:
        pred_kps_17: numpy array [N, 17, 3] in pixel coords

    Returns:
        pred_kps_15: numpy array [N, 15, 3] in pixel coords
    """
    N = pred_kps_17.shape[0]
    if N == 0:
        return np.zeros((0, 15, 3), dtype=pred_kps_17.dtype)

    pred_kps_15 = np.zeros((N, 15, 3), dtype=pred_kps_17.dtype)

    # Direct mappings: PoseTrack idx -> COCO idx
    direct_map = {
        0: 0,    # nose
        3: 5,    # left_shoulder
        4: 6,    # right_shoulder
        5: 7,    # left_elbow
        6: 8,    # right_elbow
        7: 9,    # left_wrist
        8: 10,   # right_wrist
        9: 11,   # left_hip
        10: 12,  # right_hip
        11: 13,  # left_knee
        12: 14,  # right_knee
        13: 15,  # left_ankle
        14: 16,  # right_ankle
    }

    for pt_idx, coco_idx in direct_map.items():
        pred_kps_15[:, pt_idx] = pred_kps_17[:, coco_idx]

    # Approximate head_bottom as midpoint of shoulders
    l_shoulder = pred_kps_17[:, 5]  # [N, 3]
    r_shoulder = pred_kps_17[:, 6]  # [N, 3]
    head_bottom_xy = (l_shoulder[:, :2] + r_shoulder[:, :2]) / 2
    head_bottom_vis = np.minimum(l_shoulder[:, 2], r_shoulder[:, 2])
    pred_kps_15[:, 1, :2] = head_bottom_xy
    pred_kps_15[:, 1, 2] = head_bottom_vis

    # Approximate head_top: reflect head_bottom through nose
    nose = pred_kps_17[:, 0]  # [N, 3]
    head_top_xy = 2 * nose[:, :2] - head_bottom_xy
    head_top_vis = np.minimum(nose[:, 2], head_bottom_vis)
    pred_kps_15[:, 2, :2] = head_top_xy
    pred_kps_15[:, 2, 2] = head_top_vis

    return pred_kps_15


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
                   - 'keypoints': Tensor [N, K, 3] normalised (x, y, vis)
        ann_file: path to the annotation JSON (needed for official COCO eval)
        num_keypoints: int
        keypoint_names: list[str]
        sigmas: np.array of keypoint sigmas
        kp_map_fn: callable or None — maps model's 17-kpt predictions to
                   dataset keypoints (None when dataset uses 17 COCO kpts)
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
        return dataset, ann_file, 17, COCO_KEYPOINT_NAMES, COCO_SIGMAS, None

    elif name == 'posetrack':
        ann_file = args.ann_file
        if ann_file is None:
            ann_file = os.path.join(args.data_root, 'jsons',
                                    'posetrack_val_15kpts.json')
        img_dir = args.img_dir
        if img_dir is None:
            img_dir = os.path.join(args.data_root, 'images')

        dataset = PoseTrackPoseVal(
            root=img_dir,
            annFile=ann_file,
            transforms=transforms,
            frames=args.num_frames,
            min_keypoints=args.min_kp,
        )
        return (dataset, ann_file, 15, POSETRACK_KEYPOINT_NAMES,
                POSETRACK_SIGMAS, map_coco17_to_posetrack15)

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: coco, posetrack. "
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
# Official COCO evaluation via pycocotools
# ---------------------------------------------------------------------------

def run_coco_eval(ann_file, coco_results, output_dir=None, sigmas=None):
    """Run the official COCO keypoint evaluation.

    Args:
        ann_file: path to COCO ground-truth annotation JSON
        coco_results: list of dicts in COCO keypoint result format
        output_dir: if provided, save the results JSON here
        sigmas: optional np.array of per-keypoint OKS sigmas
                (overrides pycocotools defaults; needed for non-COCO datasets)

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
    if sigmas is not None:
        coco_eval.params.kpt_oks_sigmas = sigmas
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
def evaluate(model, dataloader, postprocess, args, num_keypoints,
             sigmas, kp_map_fn=None):
    """Run inference and collect predictions in COCO result format.

    Args:
        sigmas: np.array of per-keypoint OKS sigmas
        kp_map_fn: optional callable that maps model's 17 COCO keypoint
                   predictions [N,17,3] to the dataset's keypoint format
                   [N,K,3] (e.g. PoseTrack 15 keypoints).

    Returns:
        coco_results: list of COCO-format result dicts
        num_images: number of images evaluated
        total_inference_time: total model forward-pass time in seconds
        mean_oks: mean OKS over all matched pred-GT pairs
    """
    model.eval()
    device = next(model.parameters()).device
    imgsize = (args.height, args.width)

    coco_results = []
    total_inference_time = 0.0
    num_images = 0
    total_oks = 0.0
    num_oks_matched = 0

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

            pred_boxes = result['boxes'].cpu().numpy().astype(float)
            pred_kps = result['keypoints'].cpu().numpy().astype(float)
            pred_scores = result['scores'].cpu().numpy().astype(float)

            if kp_map_fn is not None and len(pred_kps) > 0:
                pred_kps = kp_map_fn(pred_kps)

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
                        2 if float(kps[ki, 2]) > args.kp_conf_thresh else 1,
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
            gt_boxes_norm = target['boxes'].cpu().numpy().astype(float)
            gt_kps_norm = target['keypoints'].cpu().numpy().astype(float)

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
                            pred_xy = np.stack([
                                pred_kps[pi, :, 0] * scale_x,
                                pred_kps[pi, :, 1] * scale_y,
                            ], axis=1)
                            oks = compute_oks(pred_xy, gt_kps_pixel[gi], gt_areas[gi], sigmas)
                            total_oks += oks
                            num_oks_matched += 1
                            matched_gt.add(gi)
                            break

    mean_oks = total_oks / max(num_oks_matched, 1)
    return coco_results, num_images, total_inference_time, mean_oks


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
    p.add_argument('--det_tokens', type=int, default=20,
                   help='Number of detection tokens')
    p.add_argument('--pose_layers', type=int, default=2,
                   help='Number of pose decoder layers')
    p.add_argument('--num_frames', type=int, default=9,
                   help='Number of input frames (image duplicated)')

    # Dataset
    p.add_argument('--dataset', type=str, default='coco',
                   help='Dataset name (coco, posetrack)')
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

    dataset, ann_file, num_kp, kp_names, sigmas, kp_map_fn = get_dataset(args, transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
    )
    print(f"      {len(dataset)} images, {num_kp} keypoints")

    # Run inference
    print("\n[3/4] Running inference...")
    postprocess = PostProcessPose()
    coco_results, num_images, inference_time, mean_oks = evaluate(
        model, dataloader, postprocess, args, num_kp,
        sigmas=sigmas, kp_map_fn=kp_map_fn,
    )

    fps = num_images / inference_time if inference_time > 0 else 0
    print(f"      {num_images} images, {len(coco_results)} detections, {fps:.1f} img/s, mean_oks: {mean_oks:.4f}")

    # Official COCO evaluation
    print(f"\n{'='*60}")
    print("COCO Evaluation (pycocotools)")
    print(f"{'='*60}")
    coco_stats = run_coco_eval(ann_file, coco_results, args.output_dir,
                               sigmas=sigmas)

    metrics = {
        'num_images': num_images,
        'total_detections': len(coco_results),
        'fps': round(fps, 1),
        'mean_oks': round(mean_oks, 4),
        **coco_stats,
    }

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
