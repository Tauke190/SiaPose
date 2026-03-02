"""
Detection-only wrapper for SIA Pose models.

Extracts only the bounding box detection outputs from pose models,
allowing evaluation of detection performance independently using official COCO tools.

Supports:
  - sia_pose_simple
  - sia_pose_simple_dec
  - sia_pose_simple_dec_roi
"""
import os
import logging
import torch
from torch import nn

from .sia_vision_clip import VisionTransformer

logger = logging.getLogger(__name__)


class SIA_DETECTION_MODEL(nn.Module):
    """
    Detection-only wrapper for SIA models.

    Wraps an existing pose model and extracts only:
    - Predicted bounding boxes (pred_boxes)
    - Human detection scores (human_logits)

    Discard pose predictions for evaluation.
    """
    def __init__(self, pose_model):
        """
        Args:
            pose_model: A trained SIA pose model with detection head.
        """
        super(SIA_DETECTION_MODEL, self).__init__()
        self.pose_model = pose_model

    def forward(self, video):
        """
        Forward pass returning only detection outputs.

        Args:
            video: Input tensor [B, C, T, H, W] or [B, C, H, W]

        Returns:
            dict with keys:
                - 'pred_logits': [B, N, num_classes] class logits (unused for bbox detection)
                - 'pred_boxes': [B, N, 4] bounding boxes (cx, cy, w, h) normalized
                - 'human_logits': [B, N] human detection scores
        """
        with torch.no_grad():
            outputs = self.pose_model(video)

        # Extract only detection outputs
        detection_outputs = {
            'pred_logits': outputs.get('pred_logits', None),
            'pred_boxes': outputs['pred_boxes'],
            'human_logits': outputs['human_logits'],
        }

        return detection_outputs