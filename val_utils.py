"""
Shared validation utilities for SIA Pose Estimation.

Provides a single source of truth for validation logic used by both:
  - train_pose.py (during training validation)
  - val_pose.py (standalone evaluation)

This ensures consistency in metrics computation and COCO evaluation.
"""
import json
import time
import sys
import numpy as np
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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

    Args:
        ann_file: path to COCO ground-truth annotation JSON
        coco_results: list of dicts in COCO keypoint result format
        sigmas: optional np.array of per-keypoint OKS sigmas
                (overrides pycocotools defaults)

    Returns:
        dict with AP, AP50, AP75, AR, etc. stats
    """
    if len(coco_results) == 0:
        print("No predictions to evaluate.")
        return {}

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


@torch.no_grad()
def validate_pose(model, dataloader, postprocess, imgsize, num_keypoints=17,
                  sigmas=None, criterion=None, outputs=None, targets=None,
                  human_conf=0.5, keypoint_conf=0.3, kp_map_fn=None):
    """
    Run validation on a batch of data.

    This is the core validation logic shared between train_pose.py and val_pose.py.
    
    NOTE: OKS score is calculated ONLY for small+medium sized objects (area < 9216 px²)
    to focus on pose estimation performance for smaller people.

    Args:
        model: The pose estimation model (in eval mode)
        dataloader: DataLoader yielding (samples, targets) tuples
        postprocess: PostProcessPose instance for converting model outputs
        imgsize: tuple (height, width) of input images
        num_keypoints: number of keypoints (default 17 for COCO)
        sigmas: np.array of per-keypoint OKS sigmas (default COCO_SIGMAS)
        criterion: optional loss criterion for computing validation loss
        outputs: optional pre-computed model outputs (for reusing training outputs)
        targets: optional pre-computed targets (for reusing training targets)
        human_conf: human detection confidence threshold (default 0.5)
        keypoint_conf: keypoint visibility confidence threshold (default 0.3)
        kp_map_fn: optional callable to map keypoints (e.g., 17→15 for PoseTrack)

    Returns:
        dict with keys:
          - 'coco_results': list of COCO-format result dicts
          - 'num_images': total number of images processed
          - 'total_gt': total ground-truth persons
          - 'total_det': total detected persons
          - 'mean_oks': mean OKS over matched pairs (SMALL+MEDIUM objects only, area < 9216)
          - 'val_loss': validation loss (if criterion provided)
          - 'inference_time': total inference time in seconds
    """
    if sigmas is None:
        sigmas = COCO_SIGMAS

    model.eval()
    coco_results = []
    total_gt = 0
    total_det = 0
    total_val_loss = 0.0
    num_val_batches = 0
    num_images = 0
    total_oks = 0.0
    num_oks_matched = 0
    total_inference_time = 0.0
    
    # Per-component loss tracking (for detailed logging)
    per_component_losses = {}
    sigma_stats = {'sigma_min': float('inf'), 'sigma_max': float('-inf'), 'sigma_sum': 0.0, 'sigma_count': 0}

    for samples, targets_batch in tqdm(dataloader, desc="Evaluating", file=sys.stderr):
        # Move samples to device
        device = next(model.parameters()).device
        samples = samples.to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs_batch = model(samples)
            results = postprocess(outputs_batch, imgsize, human_conf=human_conf, keypoint_conf=keypoint_conf)
        total_inference_time += time.time() - t0

        # Apply keypoint mapping if needed (e.g., PoseTrack 17→15 kpts)
        if kp_map_fn is not None:
            for result in results:
                if len(result['keypoints']) > 0:
                    result['keypoints'] = kp_map_fn(result['keypoints'].cpu().numpy())
                    result['keypoints'] = torch.from_numpy(result['keypoints'])

        # Process each sample in the batch
        for result, target in zip(results, targets_batch):
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

            # Build COCO-format results
            for pi in range(len(pred_boxes)):
                box = pred_boxes[pi]
                kps = pred_kps[pi]
                kps_flat = []
                for ki in range(num_keypoints):
                    kps_flat.extend([
                        float(kps[ki, 0]) * scale_x,
                        float(kps[ki, 1]) * scale_y,
                        2 if float(kps[ki, 2]) > keypoint_conf else 0,
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

            # Compute per-instance OKS with greedy OKS-based matching
            # (matches predictions to GT based on pose similarity, not box overlap)
            gt_boxes_norm = target['boxes'].cpu().numpy().astype(float)
            gt_kps_norm = target['keypoints'].cpu().numpy().astype(float)

            if len(pred_boxes) > 0 and len(gt_boxes_norm) > 0:
                # Convert GT boxes: normalized cxcywh -> pixel xyxy (for area computation)
                gt_boxes_pixel = np.zeros_like(gt_boxes_norm)
                gt_boxes_pixel[:, 0] = (gt_boxes_norm[:, 0] - gt_boxes_norm[:, 2] / 2) * orig_w
                gt_boxes_pixel[:, 1] = (gt_boxes_norm[:, 1] - gt_boxes_norm[:, 3] / 2) * orig_h
                gt_boxes_pixel[:, 2] = (gt_boxes_norm[:, 0] + gt_boxes_norm[:, 2] / 2) * orig_w
                gt_boxes_pixel[:, 3] = (gt_boxes_norm[:, 1] + gt_boxes_norm[:, 3] / 2) * orig_h

                # Convert GT keypoints to pixel coords
                gt_kps_pixel = gt_kps_norm.copy()
                gt_kps_pixel[:, :, 0] *= orig_w
                gt_kps_pixel[:, :, 1] *= orig_h

                # GT bbox areas (w * h in pixels) - needed for OKS normalization
                gt_areas = (gt_boxes_pixel[:, 2] - gt_boxes_pixel[:, 0]) * \
                           (gt_boxes_pixel[:, 3] - gt_boxes_pixel[:, 1])

                # Convert pred keypoints to pixel coords
                pred_kps_pixel = np.zeros((len(pred_kps), num_keypoints, 2))
                pred_kps_pixel[:, :, 0] = pred_kps[:, :, 0] * scale_x
                pred_kps_pixel[:, :, 1] = pred_kps[:, :, 1] * scale_y

                # Compute OKS matrix: [num_preds, num_gts]
                num_preds = len(pred_kps_pixel)
                num_gts = len(gt_kps_pixel)
                oks_matrix = np.zeros((num_preds, num_gts))
                for pi in range(num_preds):
                    for gi in range(num_gts):
                        oks_matrix[pi, gi] = compute_oks(
                            pred_kps_pixel[pi], gt_kps_pixel[gi], gt_areas[gi], sigmas
                        )

                # Greedy matching by OKS: preds sorted by score (already sorted from postprocess)
                # FILTER: Only count OKS for small+medium objects (area < 9216 = 96²)
                matched_gt = set()
                for pi in range(num_preds):
                    oks_scores = oks_matrix[pi]
                    order = np.argsort(-oks_scores)
                    for gi in order:
                        if gi not in matched_gt and oks_scores[gi] > 0.0:
                            # Only accumulate OKS if GT is small+medium (area < 9216)
                            if gt_areas[gi] <= 9216:
                                total_oks += oks_scores[gi]
                                num_oks_matched += 1
                            matched_gt.add(gi)
                            break

        # Compute validation loss if criterion provided
        if criterion is not None:
            device = next(model.parameters()).device
            for t in targets_batch:
                t['boxes'] = t['boxes'].to(device)
                t['keypoints'] = t['keypoints'].to(device)
                t['labels'] = torch.zeros(len(t['boxes']), 1).to(device)

            loss_dict = criterion(outputs_batch, targets_batch, num_classes=1)
            # Assume weight_dict is available in criterion.weight_dict
            weight_dict = getattr(criterion, 'weight_dict', {})
            losses = sum(loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict.keys() if k in weight_dict)
            total_val_loss += losses.item()
            num_val_batches += 1
            
            # Accumulate per-component losses (weighted)
            for k in loss_dict.keys():
                if k in weight_dict:
                    component_loss = (loss_dict[k].item() * weight_dict[k])
                    if k not in per_component_losses:
                        per_component_losses[k] = 0.0
                    per_component_losses[k] += component_loss
            
            # Track RLE sigma statistics if available
            if hasattr(criterion, 'log_sigma'):
                log_sigma = criterion.log_sigma.detach().cpu()
                sigma = torch.exp(log_sigma)
                sigma_vals = sigma.numpy()
                sigma_stats['sigma_min'] = min(sigma_stats['sigma_min'], sigma_vals.min())
                sigma_stats['sigma_max'] = max(sigma_stats['sigma_max'], sigma_vals.max())
                sigma_stats['sigma_sum'] += sigma_vals.mean()
                sigma_stats['sigma_count'] += 1

    val_loss = total_val_loss / max(num_val_batches, 1) if num_val_batches > 0 else 0.0
    mean_oks = total_oks / max(num_oks_matched, 1)
    
    # Normalize per-component losses by batch count
    per_component_losses_avg = {}
    for k, v in per_component_losses.items():
        per_component_losses_avg[k] = v / max(num_val_batches, 1)
    
    # Normalize sigma stats
    sigma_stats_avg = {}
    if sigma_stats['sigma_count'] > 0:
        sigma_stats_avg['sigma_mean'] = sigma_stats['sigma_sum'] / sigma_stats['sigma_count']
        sigma_stats_avg['sigma_min'] = sigma_stats['sigma_min']
        sigma_stats_avg['sigma_max'] = sigma_stats['sigma_max']
    else:
        sigma_stats_avg = {'sigma_mean': 0.0, 'sigma_min': 0.0, 'sigma_max': 0.0}

    return {
        'coco_results': coco_results,
        'num_images': num_images,
        'total_gt': total_gt,
        'total_det': total_det,
        'mean_oks': mean_oks,
        'val_loss': val_loss,
        'num_oks_matched': num_oks_matched,
        'total_oks': total_oks,  # For distributed aggregation
        'num_val_batches': num_val_batches,  # For distributed aggregation
        'inference_time': total_inference_time,  # Total inference time in seconds
        'per_component_losses': per_component_losses_avg,  # Per-component weighted losses
        'sigma_stats': sigma_stats_avg,  # RLE sigma statistics
    }