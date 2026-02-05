"""
Image inference script for SIA model with pose estimation.
Loads image, runs inference, saves output with bounding boxes and keypoints.
"""
import os
import argparse
import numpy as np
import cv2

import torch
from torchvision.transforms import v2
from sia import get_sia, PostProcessPose, COCO_SKELETON, COCO_KEYPOINT_NAMES

# ============== DEFAULT CONFIG ==============
DEFAULT_IMAGE_PATH = "example.jpg"
DEFAULT_WEIGHT_PATH = "weights/avak_aws_stats_flt_b16_txtaug_txtlora/avak_b16_10.pt"
DEFAULT_CONF_THRESH = 0.5
DEFAULT_KP_CONF_THRESH = 0.3  # Keypoint visibility threshold
COLOR = (0, 255, 0)  # Green in BGR for bboxes
FONT = 1.0
THICKNESS = 2
DEFAULT_NUM_FRAMES = 9  # Frames per clip for model input

# Keypoint visualization colors (BGR)
KEYPOINT_COLOR = (0, 0, 255)  # Red for keypoints
SKELETON_COLOR = (255, 165, 0)  # Orange for skeleton
KEYPOINT_RADIUS = 4
SKELETON_THICKNESS = 2
# ============================================


def draw_keypoints(image, keypoints, conf_thresh=0.3):
    """
    Draw keypoints and skeleton on the image.

    Args:
        image: BGR image (H, W, 3)
        keypoints: Tensor of shape [num_keypoints, 3] -> (x, y, visibility)
        conf_thresh: Minimum visibility score to draw a keypoint

    Returns:
        Image with keypoints and skeleton drawn
    """
    keypoints = keypoints.cpu().numpy() if torch.is_tensor(keypoints) else keypoints

    # Draw skeleton connections first (so keypoints appear on top)
    for connection in COCO_SKELETON:
        # COCO_SKELETON uses 1-indexed keypoints
        idx1, idx2 = connection[0] - 1, connection[1] - 1

        if idx1 >= len(keypoints) or idx2 >= len(keypoints):
            continue

        kp1 = keypoints[idx1]
        kp2 = keypoints[idx2]

        # Only draw if both keypoints are visible
        if kp1[2] >= conf_thresh and kp2[2] >= conf_thresh:
            pt1 = (int(kp1[0]), int(kp1[1]))
            pt2 = (int(kp2[0]), int(kp2[1]))
            cv2.line(image, pt1, pt2, SKELETON_COLOR, SKELETON_THICKNESS)

    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if kp[2] >= conf_thresh:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)
            cv2.circle(image, (x, y), KEYPOINT_RADIUS, (255, 255, 255), 1)  # White outline

    return image


def parse_args():
    parser = argparse.ArgumentParser(description="SIA Image Inference Script with Pose Estimation")
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE_PATH,
                        help="Path to input image file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to output image file (default: pred_<input_name>.jpg)")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_WEIGHT_PATH,
                        help="Path to model checkpoint/weights")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., cuda:0, cpu). Default: auto-detect")
    parser.add_argument("--conf_thresh", type=float, default=DEFAULT_CONF_THRESH,
                        help="Confidence threshold for human detections")
    parser.add_argument("--kp_conf_thresh", type=float, default=DEFAULT_KP_CONF_THRESH,
                        help="Confidence threshold for keypoint visibility")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help="Number of frames for model input (image is duplicated)")
    parser.add_argument("--show", action="store_true",
                        help="Display the result image (requires display)")
    parser.add_argument("--no_pose", action="store_true",
                        help="Disable pose visualization (only show bboxes)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("\n[1] Loading SIA model...")
    model = get_sia(size='b', pretrain=None, det_token_num=20, num_frames=args.num_frames, enable_pose=True)['sia']

    if os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True), strict=False)
        print(f"    Loaded weights from: {args.checkpoint_path}")
    else:
        print(f"    No weights found at {args.checkpoint_path}, using random initialization")

    model.to(device)
    model.eval()
    print("    Model loaded!")

    # Load image
    print(f"\n[2] Loading image: {args.image_path}")
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"    ERROR: Could not load image {args.image_path}")
        return

    frame_height, frame_width = image.shape[:2]
    print(f"    Resolution: {frame_width}x{frame_height}")

    # Transforms
    tfs = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    postprocess = PostProcessPose()
    imgsize = (240, 320)
    outsize = (frame_height, frame_width)

    print(f"\n[3] Running inference...")

    # Preprocess image - duplicate to create a clip of num_frames
    resized = cv2.resize(image, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_NEAREST)
    frame_tensor = resized.transpose(2, 0, 1)  # HWC -> CHW

    # Create video clip by duplicating the frame
    clip = [frame_tensor for _ in range(args.num_frames)]
    clip_torch = torch.tensor(np.array(clip)) / 255.0
    clip_torch = tfs(clip_torch)

    # Inference
    with torch.no_grad():
        outputs = model(clip_torch.unsqueeze(0).to(device))
        result = postprocess(outputs, outsize, human_conf=args.conf_thresh, keypoint_conf=args.kp_conf_thresh)[0]

    # Draw predictions on image
    out_image = image.copy()
    boxes = result['boxes']
    keypoints = result.get('keypoints', None)

    print(f"\n[4] Results:")
    print(f"    Detected {len(boxes)} person(s)")
    print(f"    Human confidence threshold: {args.conf_thresh}")
    print(f"    Keypoint confidence threshold: {args.kp_conf_thresh}")

    for j in range(len(boxes)):
        box = boxes[j].cpu().detach().numpy().astype(int)
        cv2.rectangle(out_image, (box[0], box[1]), (box[2], box[3]), COLOR, THICKNESS)
        print(f"    Person {j+1}: bbox=({box[0]}, {box[1]}, {box[2]}, {box[3]})")

        # Draw keypoints for this person
        if keypoints is not None and len(keypoints) > j and not args.no_pose:
            kp = keypoints[j]
            num_visible = (kp[:, 2] >= args.kp_conf_thresh).sum().item() if torch.is_tensor(kp) else (kp[:, 2] >= args.kp_conf_thresh).sum()
            print(f"             keypoints: {num_visible}/17 visible")
            draw_keypoints(out_image, kp, conf_thresh=args.kp_conf_thresh)

    # Setup output path
    if args.output_path:
        output_path = args.output_path
    else:
        base_name = os.path.basename(args.image_path)
        name, ext = os.path.splitext(base_name)
        output_path = f"pred_pose_{name}.jpg"

    # Save output image
    cv2.imwrite(output_path, out_image)
    print(f"\n[5] Output saved to: {output_path}")

    # Optionally display the image
    if args.show:
        cv2.imshow("SIA Pose Prediction", out_image)
        print("    Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
