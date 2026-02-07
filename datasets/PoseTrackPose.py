"""
PoseTrack 2017 Keypoints Dataset for Pose Estimation Evaluation.

Loads PoseTrack 2017 person keypoint annotations in COCO JSON format.
PoseTrack uses 15 keypoints (vs COCO's 17): no eyes/ears, but adds head_bottom and head_top.

PoseTrack 15 keypoints:
    0: nose, 1: head_bottom, 2: head_top,
    3: left_shoulder, 4: right_shoulder,
    5: left_elbow, 6: right_elbow,
    7: left_wrist, 8: right_wrist,
    9: left_hip, 10: right_hip,
    11: left_knee, 12: right_knee,
    13: left_ankle, 14: right_ankle
"""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from util.box_ops import box_xyxy_to_cxcywh


class PoseTrackPose(Dataset):
    """PoseTrack 2017 Keypoints Dataset for pose estimation.

    Args:
        root: Path to images directory (e.g., 'posetrack_2017/images')
        annFile: Path to annotation JSON (e.g., 'posetrack_2017/jsons/posetrack_val_15kpts.json')
        transforms: Torchvision transforms to apply
        frames: Number of frames to duplicate for video input
        min_keypoints: Minimum number of visible keypoints for a person to be included
        min_area: Minimum bounding box area to include
    """

    NUM_KEYPOINTS = 15

    def __init__(
        self,
        root,
        annFile,
        transforms=None,
        frames=1,
        min_keypoints=1,
        min_area=32 * 32,
    ):
        self.root = root
        self.transforms = transforms
        self.frames = frames
        self.min_keypoints = min_keypoints
        self.min_area = min_area

        # Load annotations
        self.coco = COCO(annFile)

        # Get all image IDs with valid person annotations
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = []

        for img_id in self.coco.getImgIds(catIds=self.cat_ids):
            img_info = self.coco.loadImgs(img_id)[0]

            # Only include labeled frames
            if not img_info.get('is_labeled', True):
                continue

            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)

            # Filter: at least one person with sufficient keypoints
            valid_anns = [
                ann for ann in anns
                if ann.get('num_keypoints', 0) >= min_keypoints
                and ann.get('area', 0) >= min_area
            ]

            if len(valid_anns) > 0:
                self.img_ids.append(img_id)

        print(f"PoseTrackPose: {len(self.img_ids)} labeled images with valid person keypoints")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Load image
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size

        # Convert to tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        # Apply transforms
        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)

        # Create video clip by duplicating frame
        clip = img_tensor.unsqueeze(0).repeat(self.frames, 1, 1, 1)  # [T, C, H, W]

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        # Filter valid annotations
        valid_anns = [
            ann for ann in anns
            if ann.get('num_keypoints', 0) >= self.min_keypoints
            and ann.get('area', 0) >= self.min_area
        ]

        # Extract boxes and keypoints
        boxes = []
        keypoints = []

        for ann in valid_anns:
            # Bounding box: [x, y, w, h] -> normalize to [0, 1]
            x, y, w, h = ann['bbox']
            x1 = x / img_w
            y1 = y / img_h
            x2 = (x + w) / img_w
            y2 = (y + h) / img_h

            # Clamp to [0, 1]
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))

            # Convert to cxcywh format
            box = torch.tensor([x1, y1, x2, y2])
            box = box_xyxy_to_cxcywh(box)
            boxes.append(box)

            # Keypoints: [x0,y0,v0, x1,y1,v1, ...] -> [15, 3]
            kp = np.array(ann['keypoints']).reshape(-1, 3).astype(np.float32)
            # Normalize x, y to [0, 1]
            kp[:, 0] = kp[:, 0] / img_w
            kp[:, 1] = kp[:, 1] / img_h
            keypoints.append(torch.from_numpy(kp))

        # Stack tensors
        if len(boxes) > 0:
            boxes = torch.stack(boxes)          # [N, 4]
            keypoints = torch.stack(keypoints)  # [N, 15, 3]
        else:
            boxes = torch.zeros(0, 4)
            keypoints = torch.zeros(0, self.NUM_KEYPOINTS, 3)

        target = {
            'boxes': boxes,
            'keypoints': keypoints,
            'image_id': img_id,
        }

        return clip, target


class PoseTrackPoseVal(PoseTrackPose):
    """PoseTrack 2017 Keypoints Dataset for validation/evaluation.

    Returns original image size for proper metric computation.
    """

    def __getitem__(self, idx):
        clip, target = super().__getitem__(idx)

        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Add original image size for evaluation
        target['orig_size'] = torch.tensor([img_info['height'], img_info['width']])

        return clip, target
