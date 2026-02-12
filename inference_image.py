"""
Image inference script for SIA Pose models.
Supports all model variants: sia_pose, sia_pose_simple, sia_pose_decoder_led.
Loads image, runs inference, saves output with bounding boxes and keypoint visualization.
"""
import os
import argparse
import numpy as np
import cv2

import torch
from torchvision.transforms import v2
from sia import (
    get_sia_pose, get_sia_pose_simple, get_sia_pose_decoder_led,
    PostProcessPose, COCO_SKELETON, COCO_KEYPOINT_NAMES
)

# ============== DEFAULT CONFIG ==============
DEFAULT_IMAGE_PATH = "example.jpg"
DEFAULT_WEIGHT_PATH = "weights/model.pt"
DEFAULT_CONF_THRESH = 0.5
DEFAULT_KP_CONF_THRESH = 0.3
DEFAULT_NUM_FRAMES = 1
DEFAULT_DET_TOKENS = 20
DEFAULT_POSE_LAYERS = 2
DEFAULT_IMG_SIZE = (240, 320)  # (H, W)

# Visualization
BBOX_COLOR = (0, 255, 0)       # Green for bboxes (BGR)
BBOX_THICKNESS = 2
KEYPOINT_COLOR = (0, 0, 255)   # Red for keypoints (BGR)
SKELETON_COLOR = (255, 165, 0)  # Orange for skeleton (BGR)
KEYPOINT_RADIUS = 4
SKELETON_THICKNESS = 2
# ============================================


def draw_keypoints(image, keypoints, conf_thresh=0.3):
    """Draw keypoints and skeleton connections on the image.

    Args:
        image: BGR image (H, W, 3), modified in-place.
        keypoints: array of shape [17, 3] -> (x, y, visibility).
        conf_thresh: minimum visibility score to draw a keypoint.
    """
    kps = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints

    # Draw skeleton connections first (so keypoints appear on top)
    for connection in COCO_SKELETON:
        idx1, idx2 = connection[0] - 1, connection[1] - 1
        if idx1 >= len(kps) or idx2 >= len(kps):
            continue
        kp1, kp2 = kps[idx1], kps[idx2]
        if kp1[2] >= conf_thresh and kp2[2] >= conf_thresh:
            pt1 = (int(kp1[0]), int(kp1[1]))
            pt2 = (int(kp2[0]), int(kp2[1]))
            cv2.line(image, pt1, pt2, SKELETON_COLOR, SKELETON_THICKNESS)

    # Draw keypoints
    for kp in kps:
        if kp[2] >= conf_thresh:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)
            cv2.circle(image, (x, y), KEYPOINT_RADIUS, (255, 255, 255), 1)


def build_model(args):
    """Build the appropriate SIA pose model based on --model flag."""
    if args.size == 'b16':
        size = 'b'
    else:
        size = 'l'

    if args.model == 'sia_pose':
        model = get_sia_pose(
            size=size,
            pretrain=None,
            det_token_num=args.det_tokens,
            num_frames=args.num_frames,
            num_keypoints=17,
            pose_decoder_layers=args.pose_layers,
            enable_pose=True,
        )['sia']
    elif args.model == 'sia_pose_simple':
        model = get_sia_pose_simple(
            size=size,
            pretrain=None,
            det_token_num=args.det_tokens,
            num_frames=args.num_frames,
            num_keypoints=17,
        )['sia']
    elif args.model == 'sia_pose_decoder_led':
        model = get_sia_pose_decoder_led(
            size=size,
            pretrain=None,
            det_token_num=args.det_tokens,
            num_frames=args.num_frames,
            num_keypoints=17,
            decoder_layers=args.pose_layers,
        )['sia']
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="SIA Pose Inference Script")
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE_PATH,
                        help="Path to input image file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to output image file (default: pred_pose_<input_name>.jpg)")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_WEIGHT_PATH,
                        help="Path to model checkpoint/weights (.pt file)")
    parser.add_argument("--model", type=str, default="sia_pose",
                        choices=["sia_pose", "sia_pose_simple", "sia_pose_decoder_led"],
                        help="Model architecture variant")
    parser.add_argument("--size", type=str, default="b16", choices=["b16", "l14"],
                        help="Model size: b16 (ViT-B/16) or l14 (ViT-L/14)")
    parser.add_argument("--det_tokens", type=int, default=DEFAULT_DET_TOKENS,
                        help="Number of detection tokens (must match training)")
    parser.add_argument("--pose_layers", type=int, default=DEFAULT_POSE_LAYERS,
                        help="Number of pose decoder layers (must match training)")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help="Number of frames for model input (image is duplicated)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., cuda:0, cpu). Default: auto-detect")
    parser.add_argument("--conf_thresh", type=float, default=DEFAULT_CONF_THRESH,
                        help="Confidence threshold for human detections")
    parser.add_argument("--kp_conf_thresh", type=float, default=DEFAULT_KP_CONF_THRESH,
                        help="Confidence threshold for keypoint visibility")
    parser.add_argument("--img_height", type=int, default=DEFAULT_IMG_SIZE[0],
                        help="Model input height (must match training)")
    parser.add_argument("--img_width", type=int, default=DEFAULT_IMG_SIZE[1],
                        help="Model input width (must match training)")
    parser.add_argument("--show", action="store_true",
                        help="Display the result image (requires display)")
    parser.add_argument("--no_bbox", action="store_true",
                        help="Disable bounding box visualization")
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

    # Build model
    print(f"\n[1] Loading model: {args.model} ({args.size})")
    print(f"    det_tokens={args.det_tokens}, pose_layers={args.pose_layers}, num_frames={args.num_frames}")
    model = build_model(args)

    if os.path.exists(args.checkpoint_path):
        state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"    Loaded weights from: {args.checkpoint_path}")
    else:
        print(f"    WARNING: No weights found at {args.checkpoint_path}, using random initialization")

    model.to(device)
    model.eval()


        # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Model parameters: {total_params/1e6:.1f}M (trainable: {trainable_params/1e6:.1f}M)")
    
    if device.startswith("cuda"):
        print(f"    GPU Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"    GPU Name: {torch.cuda.get_device_name(device)}")

    print("    Model ready!")

    # Load image
    print(f"\n[2] Loading image: {args.image_path}")
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"    ERROR: Could not load image {args.image_path}")
        return

    frame_height, frame_width = image.shape[:2]
    print(f"    Resolution: {frame_width}x{frame_height}")

    # Preprocess
    imgsize = (args.img_height, args.img_width)
    tfs = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    postprocess = PostProcessPose()

    resized = cv2.resize(image, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_NEAREST)
    frame_tensor = resized.transpose(2, 0, 1)  # HWC -> CHW
    clip = [frame_tensor for _ in range(args.num_frames)]
    clip_torch = torch.tensor(np.array(clip)) / 255.0
    clip_torch = tfs(clip_torch)

    # Inference (measure only model forward pass)
    print(f"\n[3] Running inference...")
    outsize = (frame_height, frame_width)
    with torch.no_grad():
        # Move input to device before timing to exclude transfer time
        input_tensor = clip_torch.unsqueeze(0).to(device)
        print(f"    Input shape: {input_tensor.shape} (batch, frames, channels, height, width)")


        if device != "cpu":
            # GPU timing using CUDA events (accurate for asynchronous ops)
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            if args.amp:
                with torch.amp.autocast(device_type=device.split(':')[0], dtype=torch.float16):
                    outputs = model(input_tensor)
            else:
                outputs = model(input_tensor)
            end_event.record()
            torch.cuda.synchronize()
            inference_time_ms = start_event.elapsed_time(end_event)
        else:
            # CPU timing
            import time
            start_time = time.perf_counter()
            outputs = model(input_tensor)
            inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Post-process (not included in measured time)
        result = postprocess(outputs, outsize,
                             human_conf=args.conf_thresh,
                             keypoint_conf=args.kp_conf_thresh)[0]

    print(f"    Inference time: {inference_time_ms:.2f} ms")

    # Draw predictions
    out_image = image.copy()
    boxes = result['boxes']
    keypoints = result.get('keypoints', None)

    print(f"\n[4] Results:")
    print(f"    Detected {len(boxes)} person(s)")

    for j in range(len(boxes)):
        box = boxes[j].cpu().detach().numpy().astype(int)

        # Bounding box
        if not args.no_bbox:
            cv2.rectangle(out_image, (box[0], box[1]), (box[2], box[3]), BBOX_COLOR, BBOX_THICKNESS)

        # Keypoints
        if keypoints is not None and j < len(keypoints):
            kp = keypoints[j]
            num_vis = int((kp[:, 2] >= args.kp_conf_thresh).sum().item()) if torch.is_tensor(kp) else int((kp[:, 2] >= args.kp_conf_thresh).sum())
            print(f"    Person {j+1}: bbox=({box[0]}, {box[1]}, {box[2]}, {box[3]}), keypoints={num_vis}/17 visible")
            draw_keypoints(out_image, kp, conf_thresh=args.kp_conf_thresh)
        else:
            print(f"    Person {j+1}: bbox=({box[0]}, {box[1]}, {box[2]}, {box[3]})")

    # Save output
    if args.output_path:
        output_path = args.output_path
    else:
        base_name = os.path.basename(args.image_path)
        name, _ = os.path.splitext(base_name)
        output_path = f"pred_pose_{name}.jpg"

    cv2.imwrite(output_path, out_image)
    print(f"\n[5] Output saved to: {output_path}")

    # Display
    if args.show:
        cv2.imshow("SIA Pose Prediction", out_image)
        print("    Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
