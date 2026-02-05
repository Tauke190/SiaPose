"""
Image inference script for SIA model.
Loads image, runs inference, saves output with bounding boxes.
"""
import os
import argparse
import numpy as np
import cv2

import torch
from torchvision.transforms import v2
from sia import get_sia, PostProcessViz

# ============== DEFAULT CONFIG ==============
DEFAULT_IMAGE_PATH = "example.jpg"
DEFAULT_WEIGHT_PATH = "weights/avak_aws_stats_flt_b16_txtaug_txtlora/avak_b16_10.pt"
DEFAULT_CONF_THRESH = 0.5
COLOR = (0, 255, 0)  # Green in BGR
FONT = 1.0
THICKNESS = 2
DEFAULT_NUM_FRAMES = 9  # Frames per clip for model input
# ============================================


def parse_args():
    parser = argparse.ArgumentParser(description="SIA Image Inference Script")
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE_PATH,
                        help="Path to input image file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to output image file (default: pred_<input_name>.jpg)")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_WEIGHT_PATH,
                        help="Path to model checkpoint/weights")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., cuda:0, cpu). Default: auto-detect")
    parser.add_argument("--conf_thresh", type=float, default=DEFAULT_CONF_THRESH,
                        help="Confidence threshold for detections")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help="Number of frames for model input (image is duplicated)")
    parser.add_argument("--show", action="store_true",
                        help="Display the result image (requires display)")
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
    model = get_sia(size='b', pretrain=None, det_token_num=20, num_frames=args.num_frames)['sia']

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
    postprocess = PostProcessViz()
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
        result = postprocess(outputs, outsize, human_conf=args.conf_thresh)[0]

    # Draw predictions on image
    out_image = image.copy()
    boxes = result['boxes']

    print(f"\n[4] Results:")
    print(f"    Detected {len(boxes)} bounding box(es)")
    print(f"    Confidence threshold: {args.conf_thresh}")

    for j in range(len(boxes)):
        box = boxes[j].cpu().detach().numpy().astype(int)
        cv2.rectangle(out_image, (box[0], box[1]), (box[2], box[3]), COLOR, THICKNESS)
        print(f"    Box {j+1}: ({box[0]}, {box[1]}, {box[2]}, {box[3]})")

    # Setup output path
    if args.output_path:
        output_path = args.output_path
    else:
        base_name = os.path.basename(args.image_path)
        name, ext = os.path.splitext(base_name)
        output_path = f"pred_{name}.jpg"

    # Save output image
    cv2.imwrite(output_path, out_image)
    print(f"\n[5] Output saved to: {output_path}")

    # Optionally display the image
    if args.show:
        cv2.imshow("SIA Prediction", out_image)
        print("    Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
