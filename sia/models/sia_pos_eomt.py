# ============================================================================
# SIA Pose EOMT Model - Two-Stage Encoding with Learnable Pose Tokens
# ============================================================================
#
#  Based on "Your ViT is Secretly a Segmentation Model" (EOMT)
#
#  Key idea: Split the pretrained ViT into two stages. Stage 1 processes
#  spatial patches alone. Then learnable pose tokens are injected and
#  Stage 2 processes patches + tokens jointly through STANDARD encoder
#  blocks (self-attention only, no cross-attention decoder needed).
#
#  Input: Video [B, T, C, H, W]
#    ↓
#  Conv1 → Spatial patches [B*T, H*W, D]
#  + Positional Embedding + Temporal Embedding
#    ↓
#  ┌─────────────────────────────────────────────────────────────┐
#  │ STAGE 1: Pretrained Encoder Blocks (×L₁)                    │
#  │ Process spatial patches ONLY (no queries yet)               │
#  └─────────────────────────────────────────────────────────────┘
#    ↓
#  ┌─────────────────────────────────────────────────────────────┐
#  │ Inject Learnable Pose Tokens [B, N_pose, D]                 │
#  │ with Positional Embeddings (for Hungarian matching)         │
#  └─────────────────────────────────────────────────────────────┘
#    ↓
#  ┌─────────────────────────────────────────────────────────────┐
#  │ STAGE 2: Pretrained Encoder Blocks (×L₂)                    │
#  │ Standard self-attention over [patches + pose_tokens]        │
#  │ Pose tokens learn to attend to relevant spatial locations   │
#  └─────────────────────────────────────────────────────────────┘
#    ↓
#  Extract Pose Tokens [B, N_pose, D]
#    ├─→ Dedicated Pose LayerNorm
#    │
#    ├─→ DETECTION HEAD
#    │     pred_boxes [B, N_pose, 4] (cx, cy, w, h)
#    │     human_logits [B, N_pose, 2]
#    │
#    └─→ KEYPOINT REGRESSION HEAD (box-relative)
#          Keypoint embeddings [K, D] + Pose token features
#            ↓
#          Predict box-relative offsets [0,1] → convert to absolute
#          Keypoint Visibility head: D → 1
#            ↓
#          pred_keypoints [B, N_pose, K, 3]
#
# ============================================================================

"""
EOMT-based Pose Estimation Model.

Architecture from "Your ViT is Secretly a Segmentation Model":
- Split pretrained ViT into two stages
- Stage 1: Encode patches only (pretrained blocks)
- Stage 2: Encode patches + learnable pose tokens jointly (pretrained blocks)
- No decoder or cross-attention needed — queries interact with patches
  through standard self-attention in the pretrained encoder blocks
"""
import os
import logging
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..modules.sia_vision_clip import (
    VisionTransformer, MLP, ResidualAttentionBlock
)
from ..embedding_utils import interpolate_pos_embed_vit

logger = logging.getLogger(__name__)


class SIA_POSE_EOMT(nn.Module):
    """
    EOMT-based Pose Estimation Model.

    The pretrained ViT is split into two stages:
    - Stage 1 (layers 0..L₁-1): Process spatial patches only
    - Stage 2 (layers L₁..L₁+L₂-1): Process [patches + pose_tokens] jointly

    Both stages use standard self-attention encoder blocks (pretrained).
    Pose tokens learn to attend to relevant spatial features for keypoint
    localization through the standard attention mechanism.
    """

    def __init__(self,
                 size='l',
                 pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
                 pose_token_num=100,
                 num_frames=9,
                 num_keypoints=17,
                 stage2_layers=3):
        super(SIA_POSE_EOMT, self).__init__()

        # ================================================================
        # Configuration
        # ================================================================
        self.size = size.lower()
        self.pose_token_num = pose_token_num
        self.num_keypoints = num_keypoints
        self.num_frames = num_frames
        self.masking_prob = 0.0

        if self.size == 'l':
            self.patch_size = 14
            self.width = 1024
            self.total_layers = 24
            self.heads = 16
            self.output_dim = 768
            self.embed_dim = 768
        elif self.size == 'b':
            self.patch_size = 16
            self.width = 768
            self.total_layers = 12
            self.heads = 12
            self.output_dim = 512
            self.embed_dim = 512
        else:
            raise NotImplementedError(f"Size {size} not implemented")

        # Split: L₁ = total - stage2, L₂ = stage2
        self.stage2_layers = stage2_layers
        self.stage1_layers = self.total_layers - stage2_layers
        assert self.stage1_layers > 0, (
            f"stage2_layers={stage2_layers} must be < total_layers={self.total_layers}"
        )

        self.input_resolution = 224
        self.dropout_rate = 0.1

        logger.info(f'Building SIA-POSE-EOMT: size={self.size}, pose_tokens={pose_token_num}, '
                   f'num_frames={num_frames}, keypoints={num_keypoints}, '
                   f'stage1={self.stage1_layers} layers, stage2={self.stage2_layers} layers')

        logger.info("Architecture Flow:\n" +
            "  Video Input [B, C, T, H, W]\n" +
            "    ↓\n" +
            "  Vision Encoder (ViT-{:s}/{:d}) - {:d} total layers, {:d} heads, dim={:d}\n".format(
                self.size.upper(), self.patch_size, self.total_layers, self.heads, self.width) +
            "    ↓\n" +
            "  STAGE 1: Pretrained Encoder Blocks (×{:d}) — patches only\n".format(self.stage1_layers) +
            "    ↓\n" +
            "  Inject Learnable Pose Tokens={:d} + Positional Embeddings\n".format(pose_token_num) +
            "    ↓\n" +
            "  STAGE 2: Pretrained Encoder Blocks (×{:d}) — patches + pose tokens\n".format(self.stage2_layers) +
            "    ├─→ Detection Head (bboxes, human scores)\n" +
            "    └─→ Keypoint Regression Head (box-relative)\n" +
            "         └─→ {:d} keypoints XY + visibility [B, N, K, 3]\n".format(num_keypoints))

        # ================================================================
        # Module 1: Full Vision Encoder (we will manually split its blocks)
        # ================================================================
        # Build the full ViT with det_token_num=0 (we handle pose tokens ourselves)
        self.vision_encoder = VisionTransformer(
            input_resolution=self.input_resolution,
            patch_size=self.patch_size,
            width=self.width,
            layers=self.total_layers,
            heads=self.heads,
            output_dim=self.output_dim,
            kernel_size=1,
            num_frames=num_frames,
            drop_path=0.0,
            checkpoint_num=0,  # We handle checkpointing ourselves
            dropout=self.dropout_rate,
            temp_embed=True,
            det_token_num=0,  # No det tokens in encoder — we use pose tokens
        )

        # ================================================================
        # Module 2: Learnable Pose Tokens (EOMT queries)
        # ================================================================
        self.pose_token = nn.Parameter(torch.zeros(self.pose_token_num, self.width))
        nn.init.normal_(self.pose_token, std=0.02)

        # Positional embedding for pose tokens (helps Hungarian matching)
        self.pose_positional_embedding = nn.Parameter(
            (self.width ** -0.5) * torch.randn(self.pose_token_num, self.width)
        )
        nn.init.normal_(self.pose_positional_embedding, std=0.02)

        # ================================================================
        # Module 3: Dedicated LayerNorm for pose tokens
        # ================================================================
        self.pose_ln = nn.LayerNorm(self.width)

        # ================================================================
        # Module 4: Detection Heads
        # ================================================================
        self.det_head_human = self.vision_encoder.human_embed
        self.det_head_bbox = self.vision_encoder.bbox_embed

        # ================================================================
        # Module 5: Keypoint Regression Head
        # ================================================================
        self.keypoint_embed = nn.Parameter(torch.zeros(num_keypoints, self.width))
        nn.init.normal_(self.keypoint_embed, std=0.02)

        self.keypoint_xy_head = MLP(self.width, self.width, 2, 3)
        self.keypoint_vis_head = MLP(self.width, self.width // 2, 1, 2)

        # ================================================================
        # Load Pretrained Weights
        # ================================================================
        if pretrain:
            if not os.path.exists(pretrain):
                raise FileNotFoundError(f"Pretrained weights not found: {pretrain}")
            
            logger.info(f"Loading pretrained weights from {pretrain}")
            try:
                # Load directly to GPU if available, otherwise CPU
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"  Device for weight loading: {device}")
                
                checkpoint = torch.load(pretrain, map_location=device, weights_only=False)
                state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
                state_dict = interpolate_pos_embed_vit(state_dict, self.vision_encoder)
                
                # Load with strict=False to allow shape mismatches (e.g., temporal embeddings)
                missing_keys, unexpected_keys = self.vision_encoder.load_state_dict(state_dict, strict=False)
                
                # Log what was loaded
                total_keys = len(state_dict)
                loaded_keys = total_keys - len(missing_keys)
                logger.info(f"✓ Pretrained ViT loaded: {loaded_keys}/{total_keys} weights")
                
                if missing_keys:
                    logger.info(f"  Missing keys ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
                if unexpected_keys:
                    logger.info(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                
                logger.info(f"  Stage 1 (blocks[0:{self.stage1_layers}]) and Stage 2 (blocks[{self.stage1_layers}:{self.total_layers}]) both use pretrained weights")
            except Exception as e:
                raise RuntimeError(f"Failed to load pretrained weights from {pretrain}: {e}")
        else:
            logger.warning("No pretrained weights specified — model initialized randomly!")


    def no_weight_decay(self):
        """Return parameter names that should not have weight decay applied."""
        return {'pose_token', 'pose_positional_embedding', 'keypoint_embed',
                'vision_encoder.positional_embedding',
                'vision_encoder.class_embedding',
                'vision_encoder.temporal_positional_embedding'}

    def forward(self, video, masking_prob=None):
        """
        Forward pass through the EOMT pose estimation architecture.

        Args:
            video: Input video tensor [B, C, T, H, W] or [B, T, C, H, W]
            masking_prob: Optional masking probability for patches

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
        # PATCH EMBEDDING + POSITIONAL ENCODING
        # ================================================================
        x = self._embed_patches(video, masking_prob)

        # ================================================================
        # STAGE 1: Pretrained encoder blocks on patches ONLY
        # ================================================================
        x = self.vision_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # [B, N, D] → [N, B, D] for transformer

        resblocks = self.vision_encoder.transformer.resblocks
        for idx in range(self.stage1_layers):
            x = checkpoint(resblocks[idx], x, use_reentrant=False)

        # ================================================================
        # INJECT POSE TOKENS (between Stage 1 and Stage 2)
        # ================================================================
        x = x.permute(1, 0, 2)  # [N, B, D] → [B, N, D]

        pose_tokens = self.pose_token + self.pose_positional_embedding  # [N_pose, D]
        pose_tokens = pose_tokens.unsqueeze(0).expand(B, -1, -1)        # [B, N_pose, D]

        # Concatenate: [patches, pose_tokens]
        x = torch.cat((x, pose_tokens), dim=1)  # [B, N_patches + N_pose, D]

        x = x.permute(1, 0, 2)  # [B, N, D] → [N, B, D]

        # ================================================================
        # STAGE 2: Pretrained encoder blocks on [patches + pose_tokens]
        # ================================================================
        for idx in range(self.stage1_layers, self.total_layers):
            x = checkpoint(resblocks[idx], x, use_reentrant=False)

        x = self.vision_encoder.ln_post(x)
        x = x.permute(1, 0, 2)  # [N, B, D] → [B, N, D]

        # ================================================================
        # EXTRACT POSE TOKENS (with dedicated LayerNorm)
        # ================================================================
        pose_x = x[:, -self.pose_token_num:]           # [B, N_pose, D]
        pose_x = self.pose_ln(pose_x)
        pose_x = self.vision_encoder.dropout(pose_x)

        outputs = {}

        # ================================================================
        # DETECTION HEAD
        # ================================================================
        bboxes = self.det_head_bbox(pose_x).sigmoid()
        human_scores = self.det_head_human(pose_x)

        outputs['pred_boxes'] = bboxes
        outputs['human_logits'] = human_scores

        # ================================================================
        # KEYPOINT REGRESSION HEAD (box-relative)
        # ================================================================
        B_kp, num_queries, D = pose_x.shape

        # Expand learnable keypoint embeddings: [K, D] → [B, N_pose, K, D]
        kp_embed = self.keypoint_embed.unsqueeze(0).unsqueeze(0)
        kp_embed = kp_embed.expand(B_kp, num_queries, -1, -1)

        # Condition each keypoint on its parent pose token
        kp_features = kp_embed + pose_x.unsqueeze(2)  # [B, N_pose, K, D]

        kp_flat = kp_features.reshape(B_kp * num_queries * self.num_keypoints, D)

        # Predict box-relative offsets in [0,1] via sigmoid
        keypoints_rel = self.keypoint_xy_head(kp_flat).sigmoid()  # [*, 2] relative to box
        keypoints_vis = self.keypoint_vis_head(kp_flat).sigmoid()

        keypoints_rel = keypoints_rel.reshape(B_kp, num_queries, self.num_keypoints, 2)
        keypoints_vis = keypoints_vis.reshape(B_kp, num_queries, self.num_keypoints, 1)

        # Convert box-relative → absolute image coordinates
        # bboxes are [B, N_pose, 4] as (cx, cy, w, h) in [0,1]
        box_cx = bboxes[:, :, 0:1].unsqueeze(2)  # [B, N_pose, 1, 1]
        box_cy = bboxes[:, :, 1:2].unsqueeze(2)
        box_w  = bboxes[:, :, 2:3].unsqueeze(2)
        box_h  = bboxes[:, :, 3:4].unsqueeze(2)

        # keypoints_rel in [0,1] maps to box area: absolute = box_topleft + rel * box_size
        keypoints_x = box_cx + (keypoints_rel[..., 0:1] - 0.5) * box_w  # center-relative
        keypoints_y = box_cy + (keypoints_rel[..., 1:2] - 0.5) * box_h
        keypoints_xy = torch.cat([keypoints_x, keypoints_y], dim=-1)     # [B, N_pose, K, 2]

        pred_keypoints = torch.cat([keypoints_xy, keypoints_vis], dim=-1)

        outputs['pred_keypoints'] = pred_keypoints

        return outputs


    def _embed_patches(self, video, masking_prob=0.0):
        """
        Patch embedding + positional encoding (shared logic from ViT).

        Returns:
            x: Embedded patches [B, N_patches, D] ready for Stage 1
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

        # Spatial positional embedding (with interpolation for variable input sizes)
        temp_pos_embed = self.vision_encoder.positional_embedding.to(x.dtype).unsqueeze(0)
        if self.vision_encoder.positional_embedding.shape[1] != x.shape[1]:
            temp_pos_embed = self.vision_encoder.InterpolateInitPosEmbed(
                temp_pos_embed, img_size=(video.shape[-2], video.shape[-1])
            )
        x = x + temp_pos_embed

        # Temporal positional embedding
        x = x[:, 1:]  # Remove CLS token
        from einops import rearrange
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if hasattr(self.vision_encoder, 'temporal_positional_embedding'):
            if x.size(1) == 1:
                x = x + self.vision_encoder.temporal_positional_embedding.mean(1)
            else:
                x = x + self.vision_encoder.temporal_positional_embedding
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        # Optional masking (training only)
        if masking_prob > 0.0:
            x = self.vision_encoder.mask_tokens(x, masking_prob)

        return x


# ============================================================================
# Helper function for model instantiation
# ============================================================================

def get_sia_pose_eomt(size='l', pretrain=None, pose_token_num=20,
                       num_frames=1, num_keypoints=17, stage2_layers=3):
    """
    Instantiate an EOMT-based pose estimation model.

    Args:
        size: Model size ('b' for base, 'l' for large)
        pretrain: Path to pretrained weights
        pose_token_num: Number of learnable pose tokens
        num_frames: Number of input frames
        num_keypoints: Number of keypoints (17 for COCO)
        stage2_layers: Number of pretrained ViT blocks used in Stage 2 (L₂)
    """
    return SIA_POSE_EOMT(
        size=size,
        pretrain=pretrain,
        pose_token_num=pose_token_num,
        num_frames=num_frames,
        num_keypoints=num_keypoints,
        stage2_layers=stage2_layers,
    )