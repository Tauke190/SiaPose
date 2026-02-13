"""
COCO Keypoints Dataset for Pose Estimation Training.

Loads COCO person keypoint annotations and creates video-like clips
by duplicating the image for temporal consistency with the SIA model.
"""
import math
import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import ColorJitter

from util.box_ops import box_xyxy_to_cxcywh

# Left/right keypoint pairs for horizontal flip (COCO 17-keypoint format)
COCO_FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


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
        augment=False,
    ):
        self.root = root
        self.transforms = transforms
        self.frames = frames
        self.min_keypoints = min_keypoints
        self.min_area = min_area
        self.augment = augment
        self.color_jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1) if augment else None

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

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        # Filter valid annotations
        valid_anns = [
            ann for ann in anns
            if ann.get('num_keypoints', 0) >= self.min_keypoints
            and ann.get('area', 0) >= self.min_area
        ]

        # Extract boxes and keypoints in pixel coordinates
        boxes_pixel = []  # list of [x1, y1, x2, y2] in pixels
        kps_pixel = []    # list of [17, 3] arrays in pixels

        for ann in valid_anns:
            x, y, w, h = ann['bbox']
            boxes_pixel.append([x, y, x + w, y + h])

            kp = np.array(ann['keypoints']).reshape(-1, 3).astype(np.float32)
            kps_pixel.append(kp)

        # --- Augmentations (training only) ---
        if self.augment:
            # Random scale and rotation
            scale = random.uniform(0.65, 1.35)
            angle = random.uniform(-40, 40)

            if scale != 1.0 or angle != 0:
                angle_rad = math.radians(angle)
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)
                cx, cy = img_w / 2.0, img_h / 2.0

                # Inverse affine coefficients for PIL (output → input mapping)
                inv_s = 1.0 / scale
                a = cos_a * inv_s
                b = sin_a * inv_s
                c = cx - a * cx - b * cy
                d = -sin_a * inv_s
                e = cos_a * inv_s
                f = cy - d * cx - e * cy

                img = img.transform(
                    (img_w, img_h), Image.AFFINE,
                    (a, b, c, d, e, f),
                    resample=Image.BILINEAR,
                    fillcolor=(128, 128, 128),
                )

                # Forward transform keypoints
                for kp in kps_pixel:
                    for j in range(len(kp)):
                        if kp[j, 2] > 0:
                            dx, dy = kp[j, 0] - cx, kp[j, 1] - cy
                            new_x = scale * (cos_a * dx - sin_a * dy) + cx
                            new_y = scale * (sin_a * dx + cos_a * dy) + cy
                            if 0 <= new_x < img_w and 0 <= new_y < img_h:
                                kp[j, 0] = new_x
                                kp[j, 1] = new_y
                            else:
                                kp[j] = [0, 0, 0]

                # Forward transform bounding boxes
                for i in range(len(boxes_pixel)):
                    x1, y1, x2, y2 = boxes_pixel[i]
                    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    dx = corners[:, 0] - cx
                    dy = corners[:, 1] - cy
                    new_x = scale * (cos_a * dx - sin_a * dy) + cx
                    new_y = scale * (sin_a * dx + cos_a * dy) + cy
                    boxes_pixel[i] = [
                        max(0, float(new_x.min())),
                        max(0, float(new_y.min())),
                        min(img_w, float(new_x.max())),
                        min(img_h, float(new_y.max())),
                    ]

            # Random horizontal flip (p=0.5)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                for i in range(len(boxes_pixel)):
                    x1, y1, x2, y2 = boxes_pixel[i]
                    boxes_pixel[i] = [img_w - x2, y1, img_w - x1, y2]

                for kp in kps_pixel:
                    # Flip x-coordinates for labeled keypoints
                    labeled = kp[:, 2] > 0
                    kp[labeled, 0] = img_w - kp[labeled, 0]
                    # Swap left/right keypoint pairs
                    for l, r in COCO_FLIP_PAIRS:
                        kp[l], kp[r] = kp[r].copy(), kp[l].copy()

            # Color jitter
            img = self.color_jitter(img)

        # Convert to tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        # Apply transforms (resize, normalize)
        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)

        # Create video clip by duplicating frame
        clip = img_tensor.unsqueeze(0).repeat(self.frames, 1, 1, 1)  # [T, C, H, W]

        # Normalize coordinates to [0, 1] and build target
        boxes = []
        keypoints = []

        for i in range(len(boxes_pixel)):
            x1, y1, x2, y2 = boxes_pixel[i]
            # Normalize to [0, 1]
            x1 = max(0, min(1, x1 / img_w))
            y1 = max(0, min(1, y1 / img_h))
            x2 = max(0, min(1, x2 / img_w))
            y2 = max(0, min(1, y2 / img_h))

            box = torch.tensor([x1, y1, x2, y2])
            box = box_xyxy_to_cxcywh(box)
            boxes.append(box)

            kp = kps_pixel[i].copy()
            kp[:, 0] = kp[:, 0] / img_w
            kp[:, 1] = kp[:, 1] / img_h
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
