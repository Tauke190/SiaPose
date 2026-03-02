"""
Standalone evaluation script for SIA Detection models.

Evaluates bounding box detection performance only (no keypoints)
using official COCO detection metrics (AP, AP50, AP75, AR, etc.)

Supports:
  - All SIA pose model variants (simple, decoder, roi-decoder)
  - COCO dataset
  - Official COCO evaluation via pycocotools

Usage:
  python val_detection.py \
      --checkpoint weights/model.pt \
      --model sia_pose_simple \
      --data_root /path/to/coco \
      --ann_file /path/to/instances_val2017.json
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
    get_sia_pose_simple, get_sia_pose_simple_dec, get_sia_pose_coco,
)
from sia.sia_detection_model import SIA_DETECTION_MODEL
from datasets import COCOPoseVal
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCODetectionVal(torch.utils.data.Dataset):
    """Simple COCO detection dataset for evaluation (no keypoint filtering)."""

    def __init__(self, root, annFile, transforms=None, frames=9, min_area=0):
        self.root = root
        self.transforms = transforms
        self.frames = frames
        self.min_area = min_area

        self.coco = COCO(annFile)
        self.cat_ids = self.coco.getCatIds(catNms=['person'])

        # Include all images with person annotations (no keypoint filtering)
        self.img_ids = []
        for img_id in self.coco.getImgIds(catIds=self.cat_ids):
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)

            # Filter by area only
            valid_anns = [ann for ann in anns if ann.get('area', 0) >= min_area]

            if len(valid_anns) > 0:
                self.img_ids.append(img_id)

        print(f"COCODetectionVal: {len(self.img_ids)} images with person annotations")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        from PIL import Image

        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Load image
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        # Extract boxes (only need bounding boxes for detection)
        boxes = []
        for ann in anns:
            if ann.get('area', 0) >= self.min_area:
                bbox = ann['bbox']  # [x, y, w, h]
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                # Convert to normalized cxcywh
                cx = (x1 + x2) / (2 * orig_w)
                cy = (y1 + y2) / (2 * orig_h)
                bw = w / orig_w
                bh = h / orig_h
                boxes.append([cx, cy, bw, bh])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)

        # Duplicate frames for video input
        # After transforms, image is [C, H, W]
        # We need [T, C, H, W] for video input
        image = image.unsqueeze(0).repeat(self.frames, 1, 1, 1)

        target = {
            'image_id': torch.tensor(img_id, dtype=torch.long),
            'boxes': boxes,
            'orig_size': torch.tensor([orig_h, orig_w], dtype=torch.long),
        }

        return image, target


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


def run_coco_bbox_eval(ann_file, coco_results):
    """Run the official COCO detection evaluation via pycocotools.

    Args:
        ann_file: path to COCO ground-truth annotation JSON
        coco_results: list of dicts in COCO detection result format

    Returns:
        dict with AP, AP50, AP75, AR, etc. stats
    """
    if len(coco_results) == 0:
        print("No predictions to evaluate.")
        return {}

    res_path = '/tmp/sia_detection_coco_results.json'
    with open(res_path, 'w') as f:
        json.dump(coco_results, f)

    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(res_path)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    names = ['AP', 'AP50', 'AP75', 'AP_S', 'AP_M', 'AP_L',
             'AR_1', 'AR_10', 'AR_100', 'AR_S', 'AR_M', 'AR_L']
    stats = {n: float(v) for n, v in zip(names, coco_eval.stats)}
    return stats


def get_dataset(args, transforms):
    """Build the evaluation dataset for detection.

    Returns:
        dataset: COCODetectionVal dataset
        ann_file: path to the annotation JSON
    """
    ann_file = args.ann_file
    if ann_file is None:
        # Try instances annotation (detection), fall back to keypoints annotation
        ann_file = os.path.join(args.data_root, 'annotations', 'instances_val2017.json')
        if not os.path.exists(ann_file):
            # Try parent directory (in case data_root is the images subdirectory)
            parent_dir = os.path.dirname(args.data_root)
            ann_file = os.path.join(parent_dir, 'annotations', 'instances_val2017.json')
        if not os.path.exists(ann_file):
            print(f"Warning: {ann_file} not found, trying person_keypoints_val2017.json")
            ann_file = os.path.join(os.path.dirname(args.data_root), 'annotations', 'person_keypoints_val2017.json')

    img_dir = args.img_dir
    if img_dir is None:
        img_dir = os.path.join(args.data_root, 'val2017')

    dataset = COCODetectionVal(
        root=img_dir,
        annFile=ann_file,
        transforms=transforms,
        frames=args.num_frames,
        min_area=32 * 32,  # Minimum bbox area to include
    )
    return dataset, ann_file


def build_model(args):
    """Instantiate the pose model (without pretrained backbone weights)."""
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
    elif args.model == 'sia_pose_simple_dec_roi':
        model = get_sia_pose_coco(
            size=size, pretrain=None,
            det_token_num=args.det_tokens,
            num_frames=args.num_frames,
            num_keypoints=17,
            decoder_layers=args.pose_layers,
        )['sia']
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Wrap in detection-only model
    detection_model = SIA_DETECTION_MODEL(model)
    return detection_model


def collate_fn(batch):
    clips, targets = zip(*batch)
    return torch.stack(clips), list(targets)


@torch.no_grad()
def evaluate_detection(model, dataloader, args):
    """Run inference and collect predictions in COCO detection format.

    Args:
        model: Detection model (SIA_DETECTION_MODEL)
        dataloader: Evaluation dataloader
        args: Command-line arguments

    Returns:
        coco_results: list of COCO-format bbox result dicts
        num_images: number of images evaluated
        total_inference_time: total model forward-pass time
        avg_detections_per_image: average number of detections
    """
    imgsize = (args.height, args.width)
    device = next(model.parameters()).device

    coco_results = []
    num_images = 0
    total_detections = 0
    total_inference_time = 0.0

    for samples, targets in tqdm(dataloader, desc="Evaluating", file=sys.stderr):
        samples = samples.to(device)

        t0 = time.time()
        outputs = model(samples)
        total_inference_time += time.time() - t0

        for batch_idx, target in enumerate(targets):
            # Process each sample in the batch
            image_id = int(target['image_id'])
            num_images += 1

            pred_boxes = outputs['pred_boxes'][batch_idx].cpu().numpy().astype(float)
            human_scores = outputs['human_logits'][batch_idx].cpu().numpy().astype(float)

            # Ensure scores are 1D
            if human_scores.ndim > 1:
                human_scores = human_scores.squeeze()
            
            # Filter detections by score threshold
            valid_idx = human_scores > args.conf_thresh
            pred_boxes = pred_boxes[valid_idx]
            human_scores = human_scores[valid_idx]

            # Rescale from resized coords to original image coords
            orig_h, orig_w = target['orig_size'].tolist()
            scale_x = orig_w / imgsize[1]
            scale_y = orig_h / imgsize[0]

            # Convert normalized cxcywh to pixel xyxy and create COCO results
            for pi in range(len(pred_boxes)):
                box = pred_boxes[pi]  # [cx, cy, w, h] normalized

                # Denormalize
                cx = box[0] * orig_w
                cy = box[1] * orig_h
                w = box[2] * orig_w
                h = box[3] * orig_h

                # Convert to xyxy format
                x1 = cx - w / 2
                y1 = cy - h / 2

                coco_results.append({
                    'image_id': image_id,
                    'category_id': 1,  # COCO person category
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(human_scores[pi]),
                })
                total_detections += 1

    fps = num_images / total_inference_time if total_inference_time > 0 else 0
    avg_dets = total_detections / max(num_images, 1)

    return coco_results, num_images, total_inference_time, avg_dets, fps


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate SIA Detection models (bounding boxes only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to model checkpoint (.pt)')
    p.add_argument('--model', type=str, default='sia_pose_simple',
                   choices=['sia_pose_simple', 'sia_pose_simple_dec', 'sia_pose_simple_dec_roi'],
                   help='Model architecture')
    p.add_argument('--size', type=str, default='b16', choices=['b16', 'l14'],
                   help='Model size')
    p.add_argument('--det_tokens', type=int, default=20,
                   help='Number of detection tokens')
    p.add_argument('--pose_layers', type=int, default=3,
                   help='Number of pose decoder layers')
    p.add_argument('--num_frames', type=int, default=9,
                   help='Number of input frames (image duplicated)')

    # Dataset
    p.add_argument('--data_root', type=str, default=None,
                   help='Dataset root directory')
    p.add_argument('--img_dir', type=str, default=None,
                   help='Image directory (overrides data_root/val2017)')
    p.add_argument('--ann_file', type=str, default=None,
                   help='Annotation JSON path (instances_val2017.json)')

    # Input
    p.add_argument('--width', type=int, default=640, help='Input width')
    p.add_argument('--height', type=int, default=480, help='Input height')

    # Inference
    p.add_argument('--batch_size', type=int, default=32, help='Batch size')
    p.add_argument('--workers', type=int, default=0,
                   help='DataLoader workers (default 0 to avoid multiprocessing issues)')
    p.add_argument('--conf_thresh', type=float, default=0.5,
                   help='Detection confidence threshold')
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
    print(f"SIA Detection Evaluation (Bounding Boxes Only)")
    print(f"{'='*60}")
    print(f"Model       : {args.model} ({args.size})")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Device      : {device}")
    print(f"Input size  : {args.width}x{args.height}")
    print(f"Conf thresh : {args.conf_thresh}")
    print(f"{'='*60}")

    # Build model & load weights
    print("\n[1/4] Building model...")
    model = build_model(args)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    missing, unexpected = model.pose_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"      Missing keys: {missing}")
    if unexpected:
        print(f"      Unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    print(f"      Loaded weights from {args.checkpoint}")

    # Build dataset
    print("\n[2/4] Loading dataset...")
    transforms = v2.Compose([
        v2.Resize((args.height, args.width)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset, ann_file = get_dataset(args, transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True,
    )
    print(f"      {len(dataset)} images loaded")

    # Run inference
    print("\n[3/4] Running inference...")
    coco_results, num_images, inference_time, avg_dets, fps = evaluate_detection(
        model, dataloader, args
    )

    print(f"      {num_images} images, {len(coco_results)} detections")
    print(f"      Avg detections/image: {avg_dets:.2f}")
    print(f"      Inference time: {inference_time:.2f}s ({fps:.1f} img/s)")

    # Official COCO evaluation
    print(f"\n{'='*60}")
    print("COCO Detection Evaluation (pycocotools)")
    print(f"{'='*60}")
    coco_stats = run_coco_bbox_eval(ann_file, coco_results)

    metrics = {
        'num_images': num_images,
        'total_detections': len(coco_results),
        'avg_detections_per_image': round(avg_dets, 2),
        'fps': round(fps, 1),
        'inference_time_sec': round(inference_time, 2),
        **coco_stats,
    }

    # Print metrics
    print("\nDetection Metrics:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.4f}")
        else:
            print(f"  {key:30s}: {value}")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        metrics_path = os.path.join(args.output_dir, 'detection_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[4/4] Metrics saved to {metrics_path}")

        results_path = os.path.join(args.output_dir, 'coco_detection_results.json')
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