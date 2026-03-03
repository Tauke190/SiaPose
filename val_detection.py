"""
Standalone evaluation script for SIA bounding box detection on COCO.

Evaluates the bounding box localization capability of a SIA Pose model
using the official pycocotools COCO API (iouType='bbox').

Usage:
  python val_detection.py \
      --checkpoint weights/sia_ROIAlign_2.pt \
      --model sia_pose_coco \
      --dataset coco \
      --data_root /path/to/coco2017/images \
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

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from sia import (
    get_sia_pose_simple, get_sia_pose_simple_dec, get_sia_pose_coco,
    PostProcessPose,
)
from datasets import COCOPoseVal


# ---------------------------------------------------------------------------
# Model builder (mirrors val_pose.py)
# ---------------------------------------------------------------------------

def build_model(args):
    """Instantiate the model (without pretrained backbone weights)."""
    size = 'b' if args.size == 'b16' else 'l'

    if args.model == 'sia_pose_simple':
        model = get_sia_pose_simple(
            size=size, pretrain=None,
            det_token_num=args.det_tokens,
            num_frames=args.num_frames,
            num_keypoints=17,
        )['sia']
    elif args.model == 'sia_pose_simple_dec':
        model = get_sia_pose_simple_dec(
            size=size, pretrain=None,
            det_token_num=args.det_tokens,
            num_frames=args.num_frames,
            num_keypoints=17,
            decoder_layers=args.pose_layers,
        )['sia']
    elif args.model == 'sia_pose_coco':
        model = get_sia_pose_coco(
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
# COCO bbox evaluation
# ---------------------------------------------------------------------------

def run_coco_bbox_eval(ann_file, det_results):
    """Run the official COCO detection evaluation via pycocotools.

    Args:
        ann_file: path to COCO ground-truth annotation JSON
        det_results: list of dicts in COCO detection result format
                     (image_id, category_id, bbox [x,y,w,h], score)

    Returns:
        dict with AP, AP50, AP75, AP_S, AP_M, AP_L, AR1, AR10, AR100,
        AR_S, AR_M, AR_L stats
    """
    if len(det_results) == 0:
        print("No predictions to evaluate.")
        return {}

    res_path = '/tmp/sia_det_coco_results.json'
    with open(res_path, 'w') as f:
        json.dump(det_results, f)

    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(res_path)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    # Evaluate only person category
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    names = [
        'AP', 'AP50', 'AP75', 'AP_S', 'AP_M', 'AP_L',
        'AR1', 'AR10', 'AR100', 'AR_S', 'AR_M', 'AR_L',
    ]
    stats = {n: float(v) for n, v in zip(names, coco_eval.stats)}
    return stats


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def collate_fn(batch):
    clips, targets = zip(*batch)
    return torch.stack(clips), list(targets)


@torch.no_grad()
def evaluate(model, dataloader, postprocess, args):
    """Run inference and collect bbox predictions in COCO result format.

    Returns:
        det_results: list of COCO-format detection result dicts
        num_images: number of images evaluated
        total_inference_time: total model forward-pass time in seconds
    """
    imgsize = (args.height, args.width)
    device = next(model.parameters()).device

    det_results = []
    num_images = 0
    total_inference_time = 0.0

    for samples, targets in tqdm(dataloader, desc="Evaluating", file=sys.stderr):
        samples = samples.to(device)

        t0 = time.time()
        outputs = model(samples)
        results = postprocess(outputs, imgsize,
                              human_conf=args.conf_thresh,
                              keypoint_conf=0.3)
        total_inference_time += time.time() - t0

        for result, target in zip(results, targets):
            image_id = int(target['image_id'])
            num_images += 1

            pred_boxes = result['boxes'].cpu().numpy().astype(float)
            pred_scores = result['scores'].cpu().numpy().astype(float)

            # Rescale from resized coords to original image coords
            orig_h, orig_w = target['orig_size'].tolist()
            scale_x = orig_w / imgsize[1]
            scale_y = orig_h / imgsize[0]

            for pi in range(len(pred_boxes)):
                box = pred_boxes[pi]
                # Convert xyxy (resized) -> xywh (original scale)
                x1 = float(box[0]) * scale_x
                y1 = float(box[1]) * scale_y
                x2 = float(box[2]) * scale_x
                y2 = float(box[3]) * scale_y
                w = x2 - x1
                h = y2 - y1

                det_results.append({
                    'image_id': image_id,
                    'category_id': 1,
                    'bbox': [x1, y1, w, h],
                    'score': float(pred_scores[pi]),
                })

    return det_results, num_images, total_inference_time


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate SIA bounding box detection on COCO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to model checkpoint (.pt)')
    p.add_argument('--model', type=str, default='sia_pose_coco',
                   choices=['sia_pose_coco', 'sia_pose_simple', 'sia_pose_simple_dec'],
                   help='Model architecture')
    p.add_argument('--size', type=str, default='b16', choices=['b16', 'l14'],
                   help='Model size')
    p.add_argument('--det_tokens', type=int, default=20,
                   help='Number of detection tokens')
    p.add_argument('--pose_layers', type=int, default=3,
                   help='Number of pose decoder layers')
    p.add_argument('--num_frames', type=int, default=1,
                   help='Number of input frames (image duplicated)')

    # Dataset
    p.add_argument('--data_root', type=str, default=None,
                   help='Dataset root directory')
    p.add_argument('--img_dir', type=str, default=None,
                   help='Image directory (overrides data_root/val2017)')
    p.add_argument('--ann_file', type=str, default=None,
                   help='Annotation JSON path')
    p.add_argument('--min_kp', type=int, default=0,
                   help='Minimum visible keypoints per person to include')

    # Input
    p.add_argument('--width', type=int, default=640, help='Input width')
    p.add_argument('--height', type=int, default=480, help='Input height')

    # Inference
    p.add_argument('--batch_size', type=int, default=32, help='Batch size')
    p.add_argument('--workers', type=int, default=8,
                   help='DataLoader workers')
    p.add_argument('--conf_thresh', type=float, default=0.5,
                   help='Human detection confidence threshold')
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
    print(f"SIA Detection (BBox) Evaluation")
    print(f"{'='*60}")
    print(f"Model       : {args.model} ({args.size})")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Device      : {device}")
    print(f"Input size  : {args.width}x{args.height}")
    print(f"Conf thresh : {args.conf_thresh}")
    print(f"Num frames  : {args.num_frames}")
    print(f"{'='*60}")

    # Build model & load weights
    print("\n[1/4] Building model...")
    model = build_model(args)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"      Missing keys (initialized from scratch): {missing}")
    if unexpected:
        print(f"      Unexpected keys (ignored): {unexpected}")
    model.to(device)
    model.eval()
    print(f"      Loaded weights from {args.checkpoint}")

    # Build dataset
    print("\n[2/4] Loading dataset...")
    ann_file = args.ann_file
    if ann_file is None:
        ann_file = os.path.join(args.data_root, 'annotations',
                                'person_keypoints_val2017.json')
    img_dir = args.img_dir
    if img_dir is None:
        img_dir = os.path.join(args.data_root, 'val2017')

    transforms = v2.Compose([
        v2.Resize((args.height, args.width)),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = COCOPoseVal(
        root=img_dir,
        annFile=ann_file,
        transforms=transforms,
        frames=args.num_frames,
        min_keypoints=args.min_kp,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
    )
    print(f"      {len(dataset)} images")

    # Run inference
    print("\n[3/4] Running inference...")
    postprocess = PostProcessPose()
    det_results, num_images, inference_time = evaluate(
        model, dataloader, postprocess, args,
    )

    fps = num_images / inference_time if inference_time > 0 else 0
    print(f"      {num_images} images, {len(det_results)} detections, {fps:.1f} img/s")

    # Official COCO bbox evaluation
    print(f"\n{'='*60}")
    print("COCO BBox Evaluation (pycocotools, iouType='bbox')")
    print(f"{'='*60}")
    coco_stats = run_coco_bbox_eval(ann_file, det_results)

    metrics = {
        'num_images': num_images,
        'total_detections': len(det_results),
        'fps': round(fps, 1),
        **coco_stats,
    }

    # Print summary
    if coco_stats:
        print(f"\nSummary:")
        print(f"  AP       = {coco_stats.get('AP', 0):.4f}")
        print(f"  AP50     = {coco_stats.get('AP50', 0):.4f}")
        print(f"  AP75     = {coco_stats.get('AP75', 0):.4f}")
        print(f"  AP_S     = {coco_stats.get('AP_S', 0):.4f}")
        print(f"  AP_M     = {coco_stats.get('AP_M', 0):.4f}")
        print(f"  AP_L     = {coco_stats.get('AP_L', 0):.4f}")
        print(f"  AR100    = {coco_stats.get('AR100', 0):.4f}")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        metrics_path = os.path.join(args.output_dir, 'det_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[4/4] Metrics saved to {metrics_path}")

        results_path = os.path.join(args.output_dir, 'det_coco_results.json')
        with open(results_path, 'w') as f:
            json.dump(det_results, f)
        print(f"      COCO results saved to {results_path}")
    else:
        print(f"\n[4/4] Tip: use --output_dir to save results to disk")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()