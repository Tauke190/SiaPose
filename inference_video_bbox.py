"""
Headless test script for SIA model - no display required.
Loads video, runs inference, saves output video.
"""
import os
import sys
import argparse
import numpy as np
import cv2
import time

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from sia import get_sia, PostProcessViz

# ============== DEFAULT CONFIG ==============
DEFAULT_VIDEO_PATH = "example1.mp4"
DEFAULT_WEIGHT_PATH = "weights/avak_b16_11.pt"
DEFAULT_ACTIONS = ["walk", "run"]
DEFAULT_THRESH = 0.5
COLOR = (0, 255, 0)  # Green in BGR
FONT = 0.7
THICKNESS = 2
DEFAULT_NUM_FRAMES = 9  # Frames per clip for model input
# ============================================


def parse_args():
    parser = argparse.ArgumentParser(description="SIA Video Inference Script")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to config file (optional)")
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH,
                        help="Path to input video file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to output video file (default: pred_<input_name>.mp4)")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_WEIGHT_PATH,
                        help="Path to model checkpoint/weights")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., cuda:0, cpu). Default: auto-detect")
    parser.add_argument("--actions", type=str, nargs="+", default=DEFAULT_ACTIONS,
                        help="List of actions to detect")
    parser.add_argument("--thresh", type=float, default=DEFAULT_THRESH,
                        help="Detection threshold")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help="Number of frames per clip for model input")
    parser.add_argument("--show_all_scores", action="store_true",
                        help="Draw all action scores on video (even below threshold)")
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
    model = get_sia(size='b', pretrain=None, det_token_num=20, text_lora=True, num_frames=args.num_frames)['sia']

    if os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True), strict=False)
        print(f"    Loaded weights from: {args.checkpoint_path}")
    else:
        print(f"    No weights found at {args.checkpoint_path}, using random initialization")

    model.to(device)
    model.eval()
    print("    Model loaded!")

    # Open video and read all frames
    print(f"\n[2] Opening video: {args.video_path}")
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"    ERROR: Could not open video {args.video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    print(f"    Resolution: {frame_width}x{frame_height}, FPS: {fps}, Frames: {total_frames}")

    if total_frames < args.num_frames:
        print(f"    ERROR: Video has {total_frames} frames but model needs at least {args.num_frames}")
        return

    # Setup output video
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = "pred_" + os.path.basename(args.video_path).split('.')[0] + ".mp4"

    # Use temp file for OpenCV output, then convert with ffmpeg for compatibility
    temp_output_path = output_path.rsplit('.', 1)[0] + "_temp.avi"
    writer = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    print(f"    Output will be saved to: {output_path}")

    # Encode text actions
    print(f"\n[3] Encoding actions: {args.actions}")
    text_embeds = model.encode_text(args.actions)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Transforms
    tfs = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    postprocess = PostProcessViz()
    imgsize = (240, 320)
    outsize = (frame_height, frame_width)

    print(f"\n[4] Running inference...")
    print(f"    Actions: {args.actions}")
    print(f"    Threshold: {args.thresh}")

    # Process video in sliding windows
    window_size = args.num_frames * 8  # 72 frames window
    step_size = 1  # Process every frame

    for i in range(total_frames):
        # Get window of frames centered around current frame
        start_idx = max(0, i - window_size // 2)
        end_idx = min(total_frames, start_idx + window_size)
        start_idx = max(0, end_idx - window_size)  # Adjust if near end

        window_frames = frames[start_idx:end_idx]

        # If window is smaller than needed, pad with duplicates
        while len(window_frames) < window_size:
            window_frames.append(window_frames[-1])

        # Sample num_frames evenly from window
        indices = np.linspace(0, len(window_frames) - 1, args.num_frames, dtype=int)
        sampled_frames = [window_frames[idx] for idx in indices]

        # Preprocess frames
        clip = []
        for f in sampled_frames:
            resized = cv2.resize(f, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_NEAREST)
            clip.append(resized.transpose(2, 0, 1))

        clip_torch = torch.tensor(np.array(clip)) / 255.0
        clip_torch = tfs(clip_torch)

        # Inference
        with torch.no_grad():
            outputs = model.encode_vision(clip_torch.unsqueeze(0).to(device))
            outputs['pred_logits'] = F.normalize(outputs['pred_logits'], dim=-1) @ text_embeds.T

            # Get raw action scores for all persons (before thresholding)
            raw_action_scores = outputs['pred_logits'].clone()  # [1, num_detections, num_actions]

            result = postprocess(outputs, outsize, human_conf=0.9, thresh=args.thresh)[0]
            result['text_labels'] = [[args.actions[e] for e in ele] for ele in result['labels']]

        # Draw predictions on current frame
        out_frame = frames[i].copy()
        boxes = result['boxes']
        labels = result['text_labels']
        scores = result['scores']

        print(f"    Frame {i + 1}/{total_frames}: Detected {len(boxes)} persons")

        # Get all action scores for each detection
        all_action_scores = raw_action_scores[0]  # [num_detections, num_actions]

        for j in range(len(boxes)):
            box = boxes[j].cpu().detach().numpy().astype(int)
            cv2.rectangle(out_frame, (box[0], box[1]), (box[2], box[3]), COLOR, THICKNESS)

            # Get action scores for this person
            person_action_scores = all_action_scores[j] if j < all_action_scores.shape[0] else None

            # Collect actions to draw
            actions_to_draw = []
            if person_action_scores is not None:
                for action_idx, action_name in enumerate(args.actions):
                    if action_idx < person_action_scores.shape[0]:
                        score = person_action_scores[action_idx].item()
                        above_thresh = score >= args.thresh
                        if above_thresh or args.show_all_scores:
                            actions_to_draw.append((action_name, score, above_thresh))

            # Draw actions on the frame
            offset = 0
            for action_name, score, above_thresh in actions_to_draw:
                text = f"{action_name} {score:.2f}"

                # Use green for above threshold, yellow for below
                text_color = COLOR if above_thresh else (0, 255, 255)

                # Draw text with background for better visibility
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, FONT, THICKNESS
                )
                cv2.rectangle(
                    out_frame,
                    (box[0] - 2, box[1] + offset - text_height - 2),
                    (box[0] + text_width + 2, box[1] + offset + 2),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(
                    out_frame, text, (box[0], box[1] + offset),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT, text_color, THICKNESS, cv2.LINE_AA
                )
                offset += text_height + 5

        writer.write(out_frame)

        if (i + 1) % 10 == 0 or i == total_frames - 1:
            print(f"    Processed {i + 1}/{total_frames} frames")

    writer.release()

    # Convert to H.264 codec for better compatibility (VS Code, browsers, etc.)
    print(f"\n[5] Converting to H.264 for compatibility...")
    import subprocess
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_output_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            output_path
        ], check=True, capture_output=True)
        os.remove(temp_output_path)  # Remove temp file
        print(f"    Done! Output saved to: {output_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # If ffmpeg fails or not installed, keep the temp file as output
        print(f"    Warning: ffmpeg conversion failed. Using raw output.")
        if os.path.exists(temp_output_path):
            os.rename(temp_output_path, output_path)
        print(f"    Output saved to: {output_path}")

if __name__ == "__main__":
    main()
