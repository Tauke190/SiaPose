"""
SIA Pose Estimation Model - DINOv2 Backbone, Encoder-Only (Simple).

In this architecture:
- DINOv2 serves as the encoder backbone
- Detection and pose tokens are injected into DINOv2's transformer
  and participate in self-attention alongside patch tokens
- Keypoints are predicted directly from pose tokens (no decoder)

"""
import logging

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .sia_vision_dino import DINOv2Backbone, MLP

logger = logging.getLogger(__name__)


class VisionTransformerDINOSimple(nn.Module):
    """DINOv2 encoder-only model with detection and pose tokens.

    Detection and pose tokens are concatenated with DINOv2's patch tokens
    and go through all transformer blocks together (self-attention).
    Keypoints are predicted directly from pose tokens without a decoder.
    """
    def __init__(
        self,
        size='b',
        num_frames=1,
        dropout=0.,
        det_token_num=100,
        num_keypoints=17,
    ):
        super().__init__()

        self.backbone = DINOv2Backbone(size=size)
        width = self.backbone.embed_dim
        self.width = width
        self.num_frames = num_frames
        self.det_token_num = det_token_num
        self.num_keypoints = num_keypoints

        # Learned temporal positional embedding
        self.temporal_positional_embedding = nn.Parameter(
            torch.zeros(1, num_frames, width)
        )

        # Detection tokens (for bbox/human classification)
        self.det_token = nn.Parameter(torch.zeros(det_token_num, width))
        nn.init.normal_(self.det_token, std=0.02)
        self.det_positional_embedding = nn.Parameter(
            (width ** -0.5) * torch.randn(det_token_num, width)
        )
        nn.init.normal_(self.det_positional_embedding, std=0.02)

        # Pose tokens (for keypoint regression, separate from det tokens)
        self.pose_token_num = det_token_num
        self.pose_token = nn.Parameter(torch.zeros(self.pose_token_num, width))
        nn.init.normal_(self.pose_token, std=0.02)
        self.pose_positional_embedding = nn.Parameter(
            (width ** -0.5) * torch.randn(self.pose_token_num, width)
        )
        nn.init.normal_(self.pose_positional_embedding, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Keypoint projection: pose_token → num_keypoints features
        self.keypoint_proj = nn.Linear(width, num_keypoints * width)

        # Output heads
        self.human_embed = MLP(width, width, 2, 3)
        self.bbox_embed = MLP(width, width, 4, 3)
        self.keypoint_xy_head = MLP(width, width, 2, 3)
        self.keypoint_vis_head = MLP(width, width // 2, 1, 2)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'temporal_positional_embedding',
            'det_positional_embedding',
            'pose_positional_embedding',
        }

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] — video tensor (T frames)

        Returns:
            dict with keys: pred_logits, pred_boxes, human_logits, pred_keypoints
        """
        B, C, T, H, W = x.shape

        # Step 1: Reshape for per-frame DINOv2 processing
        # [B, C, T, H, W] -> [B*T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        # Step 2: Get DINOv2 prepared tokens (CLS + registers + patches with pos encoding)
        # [B*T, 1+R+N, D]
        x = self.backbone.prepare_tokens(x)
        _, S, D = x.shape  # S = 1 + num_registers + num_patches

        # Step 3: Add temporal positional embeddings
        # Reshape: [B*T, S, D] -> [B, T, S, D] -> [B*S, T, D]
        x = x.view(B, T, S, D)
        x = x.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)

        if T == 1:
            x = x + self.temporal_positional_embedding.mean(1)
        else:
            x = x + self.temporal_positional_embedding

        # [B*S, T, D] -> [B, S*T, D]
        x = x.view(B, S, T, D).permute(0, 2, 1, 3).contiguous().view(B, S * T, D)

        # Step 4: Concatenate detection and pose tokens
        det_tokens = (self.det_token + self.det_positional_embedding).unsqueeze(0).expand(B, -1, -1)
        pose_tokens = (self.pose_token + self.pose_positional_embedding).unsqueeze(0).expand(B, -1, -1)
        # Sequence: [spatial..., det_tokens..., pose_tokens...]
        x = torch.cat([x, det_tokens, pose_tokens], dim=1)

        # Step 5: Run through DINOv2 transformer blocks + LayerNorm
        x = self.backbone.forward_blocks(x)
        x = self.backbone.forward_norm(x)

        # Step 6: Extract detection and pose tokens
        # Sequence order: [spatial (S*T), det_tokens (det_token_num), pose_tokens (pose_token_num)]
        pose_x = self.dropout(x[:, -(self.pose_token_num):, :])  # [B, pose_token_num, D]
        det_x = self.dropout(x[:, -(self.det_token_num + self.pose_token_num):-(self.pose_token_num), :])  # [B, det_token_num, D]

        # Step 7: Detection output heads
        class_scores = det_x

        bboxes = checkpoint.checkpoint(
            self.bbox_embed, det_x, use_reentrant=False
        ).sigmoid()
        human_scores = checkpoint.checkpoint(
            self.human_embed, det_x, use_reentrant=False
        )

        out = {
            'pred_logits': class_scores,
            'pred_boxes': bboxes,
            'human_logits': human_scores,
        }

        # Step 8: Keypoint predictions from pose tokens
        B_kp, num_det, D_kp = pose_x.shape

        # Project pose tokens to keypoint features
        # [B, num_det, D] -> [B, num_det, num_keypoints * D]
        kp_features = self.keypoint_proj(pose_x)
        # Reshape to [B, num_det, num_keypoints, D]
        kp_features = kp_features.view(B_kp, num_det, self.num_keypoints, D_kp)

        # Flatten for MLP: [B * num_det * num_kp, D]
        kp_flat = kp_features.view(B_kp * num_det * self.num_keypoints, D_kp)

        keypoints_xy = self.keypoint_xy_head(kp_flat).sigmoid()
        keypoints_vis = self.keypoint_vis_head(kp_flat).sigmoid()

        keypoints_xy = keypoints_xy.view(B_kp, num_det, self.num_keypoints, 2)
        keypoints_vis = keypoints_vis.view(B_kp, num_det, self.num_keypoints, 1)

        pred_keypoints = torch.cat([keypoints_xy, keypoints_vis], dim=-1)
        out['pred_keypoints'] = pred_keypoints

        return out


class SIA_POSE_DINO_SIMPLE(nn.Module):
    """
    SIA Pose Model with DINOv2 backbone, encoder-only (no decoder).

    Architecture:
    - DINOv2 Encoder with detection + pose tokens in self-attention
    - Direct keypoint projection from pose tokens
    - No cross-attention decoder (simpler, faster)
    """

    def __init__(
        self,
        size='b',
        det_token_num=100,
        num_frames=1,
        num_keypoints=17,
    ):
        super().__init__()

        self.det_token_num = det_token_num
        self.num_keypoints = num_keypoints
        self.masking_prob = 0

        print(f'Type: DINOv2 Encoder-Only (SIMPLE - no decoder)')
        print(f'  - DINOv2 backbone: {DINOv2Backbone.CONFIGS[size]["model_name"]}')
        print(f'  - {det_token_num} detection + {det_token_num} pose tokens in encoder')
        print(f'  - {num_keypoints} keypoints')

        self.vision_encoder = VisionTransformerDINOSimple(
            size=size,
            num_frames=num_frames,
            dropout=0.,
            det_token_num=det_token_num,
            num_keypoints=num_keypoints,
        )

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        return ret

    def forward(self, video):
        """Forward pass for pose estimation with DINOv2 encoder-only."""
        return self.encode_vision(video)

    def encode_vision(self, image, test=False):
        """Encode image/video through DINOv2 encoder with det+pose tokens.

        Input convention (same as other SIA_POSE variants):
        - 5D: [B, T, C, H, W] -> permuted to [B, C, T, H, W]
        - 4D: [B, C, H, W] -> unsqueeze to [B, C, 1, H, W]
        """
        if image.ndim == 5:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
        else:
            image = image.unsqueeze(2)

        return self.vision_encoder(image)
