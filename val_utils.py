"""
Shared validation utilities for SIA Pose Estimation.

Provides a single source of truth for validation logic used by both:
  - train_pose.py (during training validation)
  - val_pose.py (standalone evaluation)

This ensures consistency in metrics computation and COCO evaluation.
"""
import json
import numpy as np
import torch
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
                  sigmas=None, criterion=None, outputs=None, targets=None):
    """
    Run validation on a batch of data.

    This is the core validation logic shared between train_pose.py and val_pose.py.

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

    Returns:
        dict with keys:
          - 'coco_results': list of COCO-format result dicts
          - 'num_images': total number of images processed
          - 'total_gt': total ground-truth persons
          - 'total_det': total detected persons
          - 'mean_oks': mean OKS over all matched pairs
          - 'val_loss': validation loss (if criterion provided)
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

    for samples, targets_batch in dataloader:
        # Move samples to device
        device = next(model.parameters()).device
        samples = samples.to(device)

        with torch.no_grad():
            outputs_batch = model(samples)
            results = postprocess(outputs_batch, imgsize, human_conf=0.5, keypoint_conf=0.3)

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

    val_loss = total_val_loss / max(num_val_batches, 1) if num_val_batches > 0 else 0.0
    mean_oks = total_oks / max(num_oks_matched, 1)

    return {
        'coco_results': coco_results,
        'num_images': num_images,
        'total_gt': total_gt,
        'total_det': total_det,
        'mean_oks': mean_oks,
        'val_loss': val_loss,
        'num_oks_matched': num_oks_matched,
    }