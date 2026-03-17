"""
Visualize COCO validation predictions vs ground truth.

This script loads a trained SIA Pose model, runs inference on COCO validation
images, and visualizes both predicted keypoints and ground truth annotations
side-by-side.

Usage:
    python visualize_coco.py \
        --checkpoint weights/sia_pose_coco_b16_roi_best.pt \
        --model sia_pose_coco_best \
        --num_images 50 \
        --data_root data/coco/val2017 \
        --ann_file data/coco/annotations/person_keypoints_val2017.json \
        --output_dir vis_results/
"""
import os
import argparse
import numpy as np
import cv2
import json
import random
from pathlib import Path

import torch
from torchvision.transforms import v2
from tqdm import tqdm

from sia import (
    get_sia_pose_simple, get_sia_pose_coco_decoder, get_sia_pose_coco_roi_best,
    PostProcessPose, COCO_SKELETON, COCO_KEYPOINT_NAMES
)
from datasets import COCOPoseVal
from val_utils import compute_oks, COCO_SIGMAS, box_iou_np

# Visualization colors (BGR)
PRED_BBOX_COLOR = (0, 255, 0)      # Green for predicted bboxes
GT_BBOX_COLOR = (255, 0, 0)        # Red for ground truth bboxes
PRED_KEYPOINT_COLOR = (0, 255, 0)  # Green for predicted keypoints
GT_KEYPOINT_COLOR = (255, 0, 0)    # Red for ground truth keypoints
PRED_SKELETON_COLOR = (100, 255, 100)  # Light green for predicted skeleton
GT_SKELETON_COLOR = (255, 100, 100)    # Light red for ground truth skeleton
KEYPOINT_RADIUS = 4
SKELETON_THICKNESS = 2
BBOX_THICKNESS = 2


def draw_keypoints_skeleton(image, keypoints, skeleton, color_kps, color_skeleton, 
                           conf_thresh=0.3, radius=KEYPOINT_RADIUS, thickness=SKELETON_THICKNESS):
    """Draw keypoints and skeleton connections on the image.
    
    Args:
        image: BGR image (H, W, 3), modified in-place.
        keypoints: array of shape [17, 3] -> (x, y, visibility/confidence).
        skeleton: list of (idx1, idx2) pairs for connections.
        color_kps: (B, G, R) color for keypoints.
        color_skeleton: (B, G, R) color for skeleton.
        conf_thresh: minimum score to draw a keypoint.
        radius: keypoint circle radius.
        thickness: skeleton line thickness.
    """
    kps = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints
    
    # Draw skeleton connections first (so keypoints appear on top)
    for connection in skeleton:
        idx1, idx2 = connection[0] - 1, connection[1] - 1
        if idx1 >= len(kps) or idx2 >= len(kps):
            continue
        kp1, kp2 = kps[idx1], kps[idx2]
        if kp1[2] >= conf_thresh and kp2[2] >= conf_thresh:
            pt1 = (int(kp1[0]), int(kp1[1]))
            pt2 = (int(kp2[0]), int(kp2[1]))
            cv2.line(image, pt1, pt2, color_skeleton, thickness)
    
    # Draw keypoints
    for kp in kps:
        if kp[2] >= conf_thresh:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), radius, color_kps, -1)
            cv2.circle(image, (x, y), radius, (255, 255, 255), 1)


def draw_bbox(image, bbox, color, thickness=BBOX_THICKNESS, label=""):
    """Draw bounding box on image.
    
    Args:
        image: BGR image, modified in-place.
        bbox: [x1, y1, x2, y2] in pixel coordinates.
        color: (B, G, R) color tuple.
        thickness: line thickness.
        label: optional text label.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def load_coco_gt(ann_file, img_id):
    """Load ground truth annotations for a single image.
    
    Returns:
        List of annotation dicts with keys: bbox, keypoints, num_keypoints, area.
    """
    with open(ann_file) as f:
        coco_data = json.load(f)
    
    # Find annotations for this image
    anns = [a for a in coco_data['annotations'] if a['image_id'] == img_id and a['category_id'] == 1]
    return anns


def build_model(args):
    """Build the appropriate SIA pose model based on --model flag."""
    if args.size == 'b16':
        size = 'b'
    else:
        size = 'l'

    if args.model == 'sia_pose_simple':
        model = get_sia_pose_simple(
            size=size,
            pretrain=None,
            det_token_num=args.det_tokens,
            num_frames=args.num_frames,
            num_keypoints=17,
        )['sia']
    elif args.model == 'sia_pose_coco_decoder':
        model = get_sia_pose_coco_decoder(
            size=size,
            pretrain=None,
            det_token_num=args.det_tokens,
            num_frames=args.num_frames,
            num_keypoints=17,
            decoder_layers=args.pose_layers,
        )['sia']
    elif args.model == 'sia_pose_coco_best':
        model = get_sia_pose_coco_roi_best(
            size=size,
            pretrain=None,
            det_token_num=args.det_tokens,
            num_frames=args.num_frames,
            num_keypoints=17,
            decoder_layers=args.pose_layers,
            max_roi_cap=args.max_roi_cap,
            roi_output_size=args.roi_output_size,
        )['sia']
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize COCO Pose Predictions vs Ground Truth")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint/weights (.pt file)")
    parser.add_argument("--model", type=str, default="sia_pose_coco_best",
                        choices=["sia_pose_simple", "sia_pose_coco_decoder", "sia_pose_coco_best"],
                        help="Model architecture variant")
    parser.add_argument("--size", type=str, default="b16", choices=["b16", "l14"],
                        help="Model size: b16 (ViT-B/16) or l14 (ViT-L/14)")
    parser.add_argument("--det_tokens", type=int, default=20,
                        help="Number of detection tokens (must match training)")
    parser.add_argument("--pose_layers", type=int, default=2,
                        help="Number of pose decoder layers (must match training)")
    parser.add_argument("--num_frames", type=int, default=1,
                        help="Number of frames for model input")
    parser.add_argument("--max_roi_cap", type=int, default=0,
                        help="Maximum ROI capacity (legacy)")
    parser.add_argument("--roi_output_size", type=int, default=14,
                        help="ROI output size for spatial pooling")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to COCO validation images directory")
    parser.add_argument("--ann_file", type=str, required=True,
                        help="Path to COCO annotation JSON file (person_keypoints_val2017.json)")
    parser.add_argument("--num_images", type=int, default=50,
                        help="Number of validation images to visualize")
    parser.add_argument("--output_dir", type=str, default="vis_results",
                        help="Directory to save visualization results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., cuda:0, cpu). Default: auto-detect")
    parser.add_argument("--conf_thresh", type=float, default=0.5,
                        help="Confidence threshold for human detections")
    parser.add_argument("--kp_conf_thresh", type=float, default=0.3,
                        help="Confidence threshold for keypoint visibility")
    parser.add_argument("--img_height", type=int, default=480,
                        help="Model input height")
    parser.add_argument("--img_width", type=int, default=620,
                        help="Model input width")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision (torch.amp) for faster inference")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Build model
    print(f"\n[1] Loading model: {args.model} ({args.size})")
    model = build_model(args)
    
    if os.path.exists(args.checkpoint_path):
        state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"    Loaded weights from: {args.checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    model.to(device)
    model.eval()
    
    # Load validation dataset
    print(f"\n[2] Loading COCO validation dataset")
    val_dataset = COCOPoseVal(
        root=args.data_root,
        annFile=args.ann_file,
        frames=args.num_frames,
        min_keypoints=1,
        min_area=0,
    )
    
    num_images = min(args.num_images, len(val_dataset))
    print(f"    Processing {num_images} images (total available: {len(val_dataset)})")
    
    # Create random selection of image indices
    all_indices = list(range(len(val_dataset)))
    random.shuffle(all_indices)
    selected_indices = all_indices[:num_images]
    
    # Setup transforms
    imgsize = (args.img_height, args.img_width)
    normalize = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    postprocess = PostProcessPose()
    
    # Visualization
    print(f"\n[3] Running visualization...")
    for idx in tqdm(selected_indices, desc="Visualizing"):
        # Get image ID and filename
        img_id = val_dataset.img_ids[idx]
        
        # Get image path
        coco = val_dataset.coco
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(args.data_root, img_info['file_name'])
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"    Warning: Could not load image {image_path}")
            continue
        
        frame_height, frame_width = image.shape[:2]
        
        # Prepare input for model - resize and normalize image
        resized = cv2.resize(image, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_NEAREST)
        frame_tensor = resized.transpose(2, 0, 1)  # HWC -> CHW
        
        # Create clip by duplicating frame
        clip = [frame_tensor for _ in range(args.num_frames)]
        clip_torch = torch.tensor(np.array(clip)) / 255.0  # [T, C, H, W]
        clip_torch = normalize(clip_torch)
        
        # Run inference
        with torch.no_grad():
            input_tensor = clip_torch.unsqueeze(0).to(device)
            if args.amp:
                with torch.amp.autocast(device_type=device.split(':')[0], dtype=torch.float16):
                    outputs = model(input_tensor)
            else:
                outputs = model(input_tensor)
            
            result = postprocess(outputs, (frame_height, frame_width),
                               human_conf=args.conf_thresh,
                               keypoint_conf=args.kp_conf_thresh)[0]
        
        # Create visualization
        vis_image = image.copy()
        
        # Load ground truth annotations early (needed for OKS matching in predictions loop)
        gt_anns = load_coco_gt(args.ann_file, img_id)
        
        # Draw predictions (green)
        pred_boxes = result['boxes']
        pred_keypoints = result.get('keypoints', None)
        
        pred_total_keypts = 0
        for j in range(len(pred_boxes)):
            box = pred_boxes[j].cpu().detach().numpy().astype(int)
            
            num_vis = 0
            if pred_keypoints is not None and j < len(pred_keypoints):
                kp = pred_keypoints[j]
                # Count visible keypoints for this person
                num_vis = int((kp[:, 2] >= args.kp_conf_thresh).sum().item()) if torch.is_tensor(kp) else int((kp[:, 2] >= args.kp_conf_thresh).sum())
                pred_total_keypts += num_vis
                
                # Only draw bbox and keypoints if there are visible keypoints
                if num_vis > 0:
                    draw_bbox(vis_image, box, PRED_BBOX_COLOR)
                    draw_keypoints_skeleton(vis_image, kp, COCO_SKELETON,
                                          PRED_KEYPOINT_COLOR, PRED_SKELETON_COLOR,
                                          conf_thresh=args.kp_conf_thresh)
                    
                    # Display keypoint count outside bbox
                    x1, y1, x2, y2 = box
                    text = f"KP:{num_vis}"
                    text_x = max(x1, 0)
                    text_y = max(y1 - 10, 15)  # Above the box
                    cv2.putText(vis_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PRED_KEYPOINT_COLOR, 2)
        
        # Draw ground truth (red)
        gt_total_keypts = 0
        for idx, ann in enumerate(gt_anns):
            num_gt_kpts = 0
            # Count keypoints if available
            if 'keypoints' in ann and len(ann['keypoints']) > 0:
                kps = np.array(ann['keypoints'], dtype=np.float32).reshape(17, 3)
                # Count visible GT keypoints (confidence > 0)
                num_gt_kpts = int((kps[:, 2] > 0).sum())
                gt_total_keypts += num_gt_kpts
                
                # Only draw bbox and keypoints if there are annotated keypoints
                if num_gt_kpts > 0:
                    x, y, w, h = ann['bbox']
                    bbox = [x, y, x + w, y + h]
                    draw_bbox(vis_image, bbox, GT_BBOX_COLOR)
                    draw_keypoints_skeleton(vis_image, kps, COCO_SKELETON,
                                          GT_KEYPOINT_COLOR, GT_SKELETON_COLOR,
                                          conf_thresh=0.5)  # Only draw annotated keypoints (visibility > 0)
                    
                    # Try to match with prediction and compute OKS
                    oks_score = None
                    if len(pred_boxes) > 0:
                        gt_box_xyxy = np.array([x, y, x + w, y + h])
                        pred_boxes_xyxy = pred_boxes.cpu().detach().numpy().astype(int) if torch.is_tensor(pred_boxes) else pred_boxes.astype(int)
                        ious = box_iou_np(gt_box_xyxy, pred_boxes_xyxy)
                        
                        if ious.max() > 0.1:  # Match with IoU > 0.1
                            best_pred_idx = ious.argmax()
                            if pred_keypoints is not None and best_pred_idx < len(pred_keypoints):
                                pred_kp = pred_keypoints[best_pred_idx]
                                pred_kp_xy = pred_kp[:, :2].cpu().numpy() if torch.is_tensor(pred_kp) else pred_kp[:, :2]
                                
                                # Compute OKS
                                gt_area = ann['area']
                                oks_score = compute_oks(pred_kp_xy, kps, gt_area, COCO_SIGMAS)
                    
                    # Display keypoint count outside bbox above top-left
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    text = f"KP:{num_gt_kpts}"
                    text_x = max(x1, 0)
                    text_y = max(y1 - 10, 15)  # Above the box
                    cv2.putText(vis_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GT_KEYPOINT_COLOR, 2)
                    
                    # Display OKS at the bottom-left outside bbox in red
                    if oks_score is not None:
                        oks_text = f"OKS:{oks_score:.2f}"
                        oks_x = max(x1, 0)
                        oks_y = min(y2 + 20, vis_image.shape[0] - 3)
                        cv2.putText(vis_image, oks_text, (oks_x, oks_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add legend and statistics
        y_offset = 30
        cv2.putText(vis_image, "Green: Predictions", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PRED_BBOX_COLOR, 2)
        y_offset += 40
        cv2.putText(vis_image, f"  - Detections: {len(pred_boxes)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PRED_BBOX_COLOR, 1)
        y_offset += 35
        cv2.putText(vis_image, f"  - Keypoints: {pred_total_keypts}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PRED_BBOX_COLOR, 1)
        
        y_offset += 45
        cv2.putText(vis_image, "Blue: Ground Truth", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GT_BBOX_COLOR, 2)
        y_offset += 40
        cv2.putText(vis_image, f"  - Annotations: {len(gt_anns)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GT_BBOX_COLOR, 1)
        y_offset += 35
        cv2.putText(vis_image, f"  - Keypoints: {gt_total_keypts}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GT_BBOX_COLOR, 1)
        
        # Save visualization
        output_name = f"{img_id:06d}_{Path(image_path).stem}.jpg"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), vis_image)
    
    print(f"\n[4] Visualization complete!")
    print(f"    Results saved to: {output_dir}")
    print(f"    Total images visualized: {num_images}")


if __name__ == "__main__":
    main()
