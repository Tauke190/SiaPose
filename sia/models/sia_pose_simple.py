# Encoder (ViT)
#     ↓
# 100 Detection Tokens [B, 100, D]  →  detection heads (bbox, human, class)
# 100 Pose Tokens [B, 100, D]       →  keypoint regression
#     ↓
# keypoint_proj [D → 17*D]
#     ↓
# Reshape to [B, 100, 17, D]  (17 keypoints per pose token)
#     ↓
# ┌─────────────────────────────────────┐
# │ For each keypoint embedding (D-dim) │
# ├─────────────────────────────────────┤
# │ xy_head: D→D→D→2  (x,y coords)     │
# │ vis_head: D→D/2→1  (visibility)     │
# └─────────────────────────────────────┘
#     ↓
# Output: [B, 100, 17, 3]  (x, y, visibility per keypoint)


"""
Simplified Pose Estimation Model - No Decoder.

This model attaches keypoint regression heads directly to the encoder outputs
(detection tokens) without using the cross-attention pose decoder.
"""
import os
import logging

import torch
from torch import nn
import torch.nn.functional as F

from ..modules.sia_vision_clip import (
    VisionTransformer, MLP, inflate_weight, load_state_dict,
    _MODELS, PoseDecoderLayer
)
from ..embedding_utils import interpolate_pos_embed_vit

logger = logging.getLogger(__name__)


# ============================================================================
# SIA Pose Model - Simple Direct Keypoint Regression (Composition-based)
# ============================================================================

class SIA_POSE_SIMPLE(nn.Module):
    """
    Simple SIA Pose Model with direct keypoint regression using composition.
    
    Architecture:
    1. Vision Encoder: ViT processes [patches + det_tokens + pose_tokens]
    2. Detection Head: det_tokens → bbox and human predictions
    3. Keypoint Head: pose_tokens → keypoint coordinates directly (no decoder)
    
    No pose decoder needed - faster and simpler than decoder-based approaches.
    """
    
    def __init__(self,
                 size='l',
                 pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
                 det_token_num=100,
                 num_frames=9,
                 num_keypoints=17):
        super(SIA_POSE_SIMPLE, self).__init__()

        # ================================================================
        # Configuration
        # ================================================================
        self.size = size.lower()
        self.det_token_num = det_token_num
        self.num_keypoints = num_keypoints
        self.num_frames = num_frames
        self.masking_prob = 0.0
        
        # Model parameters based on size
        if self.size == 'l':
            self.patch_size = 14
            self.width = 1024
            self.layers = 24
            self.heads = 16
            self.output_dim = 768
            self.embed_dim = 768
        elif self.size == 'b':
            self.patch_size = 16
            self.width = 768
            self.layers = 12
            self.heads = 12
            self.output_dim = 512
            self.embed_dim = 512
        else:
            raise NotImplementedError(f"Size {size} not implemented")

        self.input_resolution = 224
        self.dropout_rate = 0.1
        
        logger.info(f'Building SIA-POSE-SIMPLE: size={self.size}, det_tokens={det_token_num}, '
                   f'num_frames={num_frames}, keypoints={num_keypoints}')

        # Print architecture diagram
        logger.info("Architecture Flow:\n" +
            "  Video Input [B, C, T, H, W]\n" +
            "    ↓\n" +
            "  Vision Encoder (ViT-{:s}/{:d}) - {:d} layers, {:d} heads, dim={:d}\n".format(
                self.size.upper(), self.patch_size, self.layers, self.heads, self.width) +
            "    ↓\n" +
            "  [Spatial Features] + [DET={:d}] + [POSE={:d}] Tokens\n".format(det_token_num, det_token_num) +
            "    ├─→ Detection Head (bboxes, human scores, class logits)\n" +
            "    └─→ Keypoint Head (direct regression, no decoder)\n" +
            "         └─→ {:d} keypoints XY + visibility [B, N, K, 3]\n".format(num_keypoints) +
            "  Output: pred_boxes, human_logits, pred_keypoints")

        # ================================================================
        # Module 1: Vision Encoder (ViT)
        # ================================================================
        self.vision_encoder = VisionTransformer(
            input_resolution=self.input_resolution,
            patch_size=self.patch_size,
            width=self.width,
            layers=self.layers,
            heads=self.heads,
            output_dim=self.output_dim,
            kernel_size=1,
            num_frames=num_frames,
            drop_path=0.0,
            checkpoint_num=24 if self.size == 'l' else 12,
            dropout=self.dropout_rate,
            temp_embed=True,
            det_token_num=det_token_num,
        )

        # ================================================================
        # Module 2: Learnable Pose Tokens (created before encoder)
        # ================================================================
        self.pose_token_num = det_token_num
        self.pose_token = nn.Parameter(torch.zeros(self.pose_token_num, self.width))
        nn.init.normal_(self.pose_token, std=0.02)
        
        # Separate positional embedding for pose tokens (different from det tokens)
        self.pose_positional_embedding = nn.Parameter(
            (self.width ** -0.5) * torch.randn(self.pose_token_num, self.width)
        )
        nn.init.normal_(self.pose_positional_embedding, std=0.02)

        # ================================================================
        # Module 3: Detection Heads (for bounding boxes)
        # ================================================================
        self.det_head_human = self.vision_encoder.human_embed
        self.det_head_bbox = self.vision_encoder.bbox_embed
        self.det_head_class = self.vision_encoder.proj

        # ================================================================
        # Module 4: Keypoint Heads (for pose estimation)
        # ================================================================
        # Learnable per-keypoint embeddings (efficient alternative to heavy projection)
        self.keypoint_embed = nn.Parameter(torch.zeros(num_keypoints, self.width))
        nn.init.normal_(self.keypoint_embed, std=0.02)

        self.keypoint_xy_head = MLP(self.width, self.width, 2, 3)
        self.keypoint_vis_head = MLP(self.width, self.width // 2, 1, 2)

        # ================================================================
        # Load Pretrained Weights
        # ================================================================
        if pretrain and os.path.exists(pretrain):
            logger.info(f"Loading pretrained weights from {pretrain}")
            state_dict = torch.load(pretrain, map_location='cpu', weights_only=False)['model']
            state_dict = interpolate_pos_embed_vit(state_dict, self.vision_encoder)
            self.vision_encoder.load_state_dict(state_dict, strict=False)


    def no_weight_decay(self):
        """Return parameter names that should not have weight decay applied."""
        return {'pose_token', 'pose_positional_embedding', 'keypoint_embed',
                'vision_encoder.positional_embedding',
                'vision_encoder.class_embedding',
                'vision_encoder.temporal_positional_embedding'}

    def forward(self, video, masking_prob=None):
        """
        Forward pass through the pose estimation architecture.
        
        Args:
            video: Input video tensor
            masking_prob: Optional masking probability
            
        Returns:
            Dictionary with detection and pose predictions
        """
        if masking_prob is None:
            masking_prob = self.masking_prob
            
        # Normalize video format
        if video.ndim == 5:
            video = video.permute(0, 2, 1, 3, 4).contiguous()
        else:
            video = video.unsqueeze(2)

        B = video.shape[0]
        
        # ================================================================
        # ENCODER STAGE: Vision Transformer processes all tokens
        # ================================================================
        encoder_out = self._forward_encoder(video, masking_prob)
        
        spatial_features = encoder_out['spatial_features']
        det_x = encoder_out['det_tokens']
        pose_x = encoder_out['pose_tokens']

        outputs = {}

        # ================================================================
        # DETECTION HEAD STAGE
        # ================================================================
        if self.det_head_class is not None:
            class_logits = det_x @ self.det_head_class
        else:
            class_logits = det_x
            
        bboxes = self.det_head_bbox(det_x).sigmoid()
        human_scores = self.det_head_human(det_x)
        
        outputs['pred_logits'] = class_logits
        outputs['pred_boxes'] = bboxes
        outputs['human_logits'] = human_scores

        # ================================================================
        # KEYPOINT HEAD STAGE: Direct regression from pose tokens
        # ================================================================
        if pose_x is not None:
            B_kp, num_queries, D = pose_x.shape

            # Expand learnable keypoint embeddings: [num_keypoints, D] → [B, num_queries, num_keypoints, D]
            # Each of the 17 keypoints has its own learned embedding
            kp_embed = self.keypoint_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, num_keypoints, D]
            kp_embed = kp_embed.expand(B_kp, num_queries, -1, -1)    # [B, num_queries, num_keypoints, D]

            # Condition each keypoint on its parent detection's pose features
            kp_features = kp_embed + pose_x.unsqueeze(2)  # [B, num_queries, num_keypoints, D]

            kp_flat = kp_features.view(B_kp * num_queries * self.num_keypoints, D)

            # Predict keypoint coordinates using shared MLP
            keypoints_xy = self.keypoint_xy_head(kp_flat).sigmoid()
            # Predict keypoint visibility using shared MLP
            keypoints_vis = self.keypoint_vis_head(kp_flat).sigmoid()

            keypoints_xy = keypoints_xy.view(B_kp, num_queries, self.num_keypoints, 2)
            keypoints_vis = keypoints_vis.view(B_kp, num_queries, self.num_keypoints, 1)
            pred_keypoints = torch.cat([keypoints_xy, keypoints_vis], dim=-1)

            outputs['pred_keypoints'] = pred_keypoints

        return outputs


    def _forward_encoder(self, video, masking_prob=0.0):
        """
        Extract features from the vision encoder.
        
        The encoder processes [CLS + spatial_patches] combined with
        [detection tokens] and [pose tokens] to produce enriched features.
        
        Returns:
            Dictionary with spatial_features and token outputs
        """
        x = self.vision_encoder.conv1(video)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        # Add [CLS] token and spatial positional embedding
        x = torch.cat([
            self.vision_encoder.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        
        # Spatial positional embedding
        temp_pos_embed = self.vision_encoder.positional_embedding.to(x.dtype).unsqueeze(0)
        if self.vision_encoder.positional_embedding.shape[1] != x.shape[1]:
            temp_pos_embed = self.vision_encoder.InterpolateInitPosEmbed(
                temp_pos_embed, img_size=(video.shape[-2], video.shape[-1])
            )
        x = x + temp_pos_embed

        # Temporal positional embedding
        x = x[:, 1:]
        from einops import rearrange
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if hasattr(self.vision_encoder, 'temporal_positional_embedding'):
            if x.size(1) == 1:
                x = x + self.vision_encoder.temporal_positional_embedding.mean(1)
            else:
                x = x + self.vision_encoder.temporal_positional_embedding
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        # Optional masking
        if masking_prob > 0.0:
            x = self.vision_encoder.mask_tokens(x, masking_prob)

        # ================================================================
        # Add [DET] and [POSE] tokens before encoder
        # ================================================================
        if self.det_token_num > 0:
            # Detection tokens
            det_tokens = self.vision_encoder.det_token + self.vision_encoder.det_positional_embedding
            det_tokens = det_tokens.unsqueeze(0).expand(B, -1, -1)
            
            # Pose tokens (with separate positional embedding)
            pose_tokens = self.pose_token + self.pose_positional_embedding
            pose_tokens = pose_tokens.unsqueeze(0).expand(B, -1, -1)
            
            # Concatenate: spatial_patches + det_tokens + pose_tokens
            x = torch.cat((x, det_tokens, pose_tokens), dim=1)
            num_pose_tokens = self.pose_token_num
        else:
            num_pose_tokens = 0

        # Forward through transformer
        x = self.vision_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.vision_encoder.transformer(x)
        x = self.vision_encoder.ln_post(x)
        x = x.permute(1, 0, 2)

        # ================================================================
        # Split outputs into spatial features and token outputs
        # ================================================================
        if self.det_token_num > 0:
            spatial_features = x[:, :-self.det_token_num-num_pose_tokens]
            det_x = self.vision_encoder.dropout(x[:, -self.det_token_num-num_pose_tokens:-num_pose_tokens])
            pose_x = self.vision_encoder.dropout(x[:, -num_pose_tokens:])
        else:
            spatial_features = x
            pose_x = None
            det_x = None

        return {
            'spatial_features': spatial_features,
            'det_tokens': det_x,
            'pose_tokens': pose_x,
        }



