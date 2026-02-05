"""
COCO Keypoints Dataset for Pose Estimation Training.

Loads COCO person keypoint annotations and creates video-like clips
by duplicating the image for temporal consistency with the SIA model.
"""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from util.box_ops import box_xyxy_to_cxcywh


class COCOPose(Dataset):
    """COCO Keypoints Dataset for pose estimation.

    Args:
        root: Path to COCO images directory (e.g., 'coco/train2017')
        annFile: Path to annotation file (e.g., 'coco/annotations/person_keypoints_train2017.json')
        transforms: Torchvision transforms to apply
        frames: Number of frames to duplicate for video input
        min_keypoints: Minimum number of visible keypoints for a person to be included
        min_area: Minimum bounding box area to include
    """

    def __init__(
        self,
        root,
        annFile,
        transforms=None,
        frames=9,
        min_keypoints=5,
        min_area=32 * 32,
    ):
        self.root = root
        self.transforms = transforms
        self.frames = frames
        self.min_keypoints = min_keypoints
        self.min_area = min_area

        # Load COCO annotations
        self.coco = COCO(annFile)

        # Get all image IDs that have person annotations with keypoints
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = []

        for img_id in self.coco.getImgIds(catIds=self.cat_ids):
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

        print(f"COCOPose: {len(self.img_ids)} images with valid person keypoints")

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

        # Apply transforms (resize, normalize)
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
            # Convert to [x1, y1, x2, y2] normalized
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

            # Keypoints: [x1, y1, v1, x2, y2, v2, ...] -> [17, 3]
            kp = np.array(ann['keypoints']).reshape(-1, 3).astype(np.float32)
            # Normalize x, y to [0, 1]
            kp[:, 0] = kp[:, 0] / img_w  # x
            kp[:, 1] = kp[:, 1] / img_h  # y
            # visibility: 0=not labeled, 1=labeled but occluded, 2=labeled and visible
            # Keep as is for loss computation
            keypoints.append(torch.from_numpy(kp))

        # Stack tensors
        if len(boxes) > 0:
            boxes = torch.stack(boxes)  # [N, 4]
            keypoints = torch.stack(keypoints)  # [N, 17, 3]
        else:
            boxes = torch.zeros(0, 4)
            keypoints = torch.zeros(0, 17, 3)

        target = {
            'boxes': boxes,
            'keypoints': keypoints,
            'image_id': img_id,
        }

        return clip, target


class COCOPoseVal(COCOPose):
    """COCO Keypoints Dataset for validation/evaluation.

    Returns original image size for proper metric computation.
    """

    def __getitem__(self, idx):
        clip, target = super().__getitem__(idx)

        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Add original image size for evaluation
        target['orig_size'] = torch.tensor([img_info['height'], img_info['width']])

        return clip, target
