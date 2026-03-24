# ============================================================================
# ROI Decoder variant for Pose Estimation
# ============================================================================
#
# ACTUAL Architecture (vs what's commented below):
#   1. Encoder (ViT): [image_patches + det_tokens + pose_tokens]
#      → spatial_features [B, H*W*T, D] + det_x [B, N, D] + pose_x [B, N, D]
#      NOTE: Pose tokens ARE processed through the full 12-layer encoder!
#
#   2. Detection Head: det_x → bboxes [B, N, 4] (cx, cy, w, h), human scores
#
#   3. ROI Extraction: spatial_features + predicted_bboxes → roi_features [B, N, P*P, D]
#      where P=14 (so 196 tokens per person)
#      NOTE: Bboxes are DETACHED - no gradients flow back to detection head
#
#   4. Pose Decoder (3 layers):
#      - Self-attn: all N pose queries interact globally
#      - Cross-attn: each query attends ONLY to its own ROI (196 patches)
#      - Each person gets isolated context window of 196 patches vs 1200 in decoder model
#
#   5. Pose Head: pose_tokens → keypoint offsets relative to bbox
#      - Predicts relative offsets [-1, 1] per keypoint
#      - Converts to global coords: kp = bbox_center + offset * bbox_size
#
# Key difference from SIA_POSE_SIMPLE_DEC:
#   - ROI model: limited to 196 patches per person (14x14 ROI)
#   - Decoder model: attends to all 1200 patches (full image context)
#   - ROI model: keypoints anchored to predicted bboxes (indirect)
#   - Decoder model: direct global coordinate prediction (no bbox dependency)
#   - ROI model: bbox predictions not supervised through pose loss (.detach())
#   - Decoder model: no bbox-keypoint coupling issues
#
# ============================================================================

import os
import logging

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from ..modules.sia_vision_clip import (
    VisionTransformer, MLP, inflate_weight, load_state_dict,
    _MODELS, PoseDecoderLayer
)
from ..embedding_utils import interpolate_temporal_pos_embed, load_temp_embed_with_mismatch, interpolate_pos_embed_vit

logger = logging.getLogger(__name__)


def extract_roi_features_aligned(spatial_features, bboxes, H_patches, W_patches,
                                  num_frames=1, output_size=14, padding=0.1):
    """
    Extract fixed-size ROI features using torchvision.ops.roi_align.

    Replaces the Python for-loop with a single CUDA-optimized call.
    Fixed output size → no padding → no mask → FlashAttention enabled.

    Args:
        spatial_features: [B, H*W*T, D] encoder spatial output
        bboxes:          [B, N, 4] predicted boxes in (cx, cy, w, h) normalized format
        H_patches:       int, number of patch rows
        W_patches:       int, number of patch columns
        num_frames:      int, number of temporal frames
        output_size:     int, P where each ROI is pooled to PxP tokens
        padding:         float, fractional padding around bbox

    Returns:
        roi_features: [B, N, P*P, D] fixed-size ROI features per detection
        roi_mask:     None (no padding needed)
        roi_stats:    dict with ROI statistics
    """
    B, N, _ = bboxes.shape
    D = spatial_features.shape[-1]
    device = spatial_features.device
    dtype = spatial_features.dtype

    # Temporal averaging (same as original)
    if num_frames > 1:
        spatial_2d = spatial_features.reshape(
            B, H_patches * W_patches, num_frames, D
        ).mean(dim=2)
    else:
        spatial_2d = spatial_features  # [B, H*W, D]

    # Reshape to NCHW format for roi_align: [B, D, H_patches, W_patches]
    feat_map = spatial_2d.reshape(B, H_patches, W_patches, D).permute(0, 3, 1, 2)

    # Convert normalized (cx, cy, w, h) → absolute patch-space (x1, y1, x2, y2)
    cx = bboxes[:, :, 0]
    cy = bboxes[:, :, 1]
    bw = bboxes[:, :, 2] * (1 + 2 * padding)
    bh = bboxes[:, :, 3] * (1 + 2 * padding)

    x1 = ((cx - bw / 2) * W_patches).clamp(min=0)
    y1 = ((cy - bh / 2) * H_patches).clamp(min=0)
    x2 = ((cx + bw / 2) * W_patches).clamp(max=W_patches)
    y2 = ((cy + bh / 2) * H_patches).clamp(max=H_patches)

    # Build roi_align input: [B*N, 5] = (batch_idx, x1, y1, x2, y2)
    batch_idx = torch.arange(B, device=device, dtype=feat_map.dtype
                             ).unsqueeze(1).expand(B, N).reshape(-1, 1)
    rois = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4).to(feat_map.dtype)
    rois = torch.cat([batch_idx, rois], dim=1)  # [B*N, 5]

    # roi_align: single CUDA kernel, bilinear interpolation
    # Input: feat_map [B, D, H, W], Output: [B*N, D, P, P]
    roi_out = roi_align(
        feat_map.float(),  # roi_align requires float32
        rois.float(),
        output_size=(output_size, output_size),
        spatial_scale=1.0,  # features already in patch-space coordinates
        aligned=True,
    )

    # Convert back to original dtype
    if dtype != torch.float32:
        roi_out = roi_out.to(dtype)

    # Reshape to [B, N, P*P, D]
    P = output_size
    roi_features = roi_out.reshape(B, N, D, P * P).permute(0, 1, 3, 2)

    # Statistics for logging (all fixed-size, zero waste)
    roi_stats = {
        'roi_count': B * N,
        'roi_mean': float(P * P),
        'roi_std': 0.0,
        'roi_min': P * P,
        'roi_max': P * P,
        'roi_median': float(P * P),
        'max_roi_len': P * P,
        'capped_count': 0,
        'padding_waste_pct': 0.0,
        'total_slots': B * N * P * P,
        'valid_patches': B * N * P * P,
    }

    return roi_features, None, roi_stats


# ROI Pose Decoder Layer
class ROIPoseDecoderLayer(nn.Module):
    """
    Decoder layer where each pose query cross-attends to its own ROI features.

    Self-attention: all N pose queries interact with each other (global context).
    Cross-attention: each query_i attends only to roi_features_i (local ROI).
        When no padding exists (roi_mask=None), nn.MHA uses FlashAttention via SDPA.
    FFN: standard feed-forward.
    """
    def __init__(self, d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1):
        super().__init__()
        # Self-attention among all pose queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention: each query attends to its own ROI
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries, roi_features, roi_mask=None):
        """
        Args:
            queries:      [B, N, D]  pose queries (one per detection)
            roi_features: [B, N, roi_len, D]  per-detection ROI spatial features
            roi_mask:     [B, N, roi_len] bool, True = padding (to be ignored).
                          None when no padding exists (enables FlashAttention).

        Returns:
            queries: [B, N, D] refined pose queries
        """
        B, N, D = queries.shape
        roi_len = roi_features.shape[2]

        # --- Self-attention (global, across all N queries) ---
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]

        # --- Cross-attention (per-detection, each query -> its ROI) ---
        q2 = self.norm2(queries)
        q_flat = q2.reshape(B * N, 1, D)
        kv_flat = roi_features.reshape(B * N, roi_len, D)
        mask_flat = roi_mask.reshape(B * N, roi_len) if roi_mask is not None else None

        cross_out = self.cross_attn(
            query=q_flat,
            key=kv_flat,
            value=kv_flat,
            key_padding_mask=mask_flat,
        )[0]  # [B*N, 1, D]

        queries = queries + cross_out.reshape(B, N, D)

        # --- FFN ---
        queries = queries + self.ffn(self.norm3(queries))
        return queries

# Vision Transformer with ROI Decoder
# ============================================================================
# SIA Pose Model with ROI-based Pose Decoder (Composition-based)
# ============================================================================

class SIA_POSE_SIMPLE_DEC_ROI(nn.Module):
    """
    SIA Pose Model with ROI-based pose decoder using composition architecture.
    
    Architecture:
    1. Vision Encoder: ViT processes [patches + det_tokens + pose_tokens]
    2. Detection Head: det_tokens → bbox and human predictions
    3. ROI Extraction: Use roi_align to extract fixed-size ROI patches from spatial features
    4. Pose Decoder: pose_tokens + cross-attention to ROI features
    5. Keypoint Head: pose_tokens → keypoint offsets relative to bbox center
    
    Key difference from decoder-based: Only attends to ROI features (patches inside 
    detected bboxes), not all spatial patches. This focuses computation on relevant regions.
    """
    
    def __init__(self,
                 size='l',
                 pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
                 det_token_num=100,
                 num_frames=9,
                 num_keypoints=17,
                 decoder_layers=3,
                 roi_output_size=14):
        super(SIA_POSE_SIMPLE_DEC_ROI, self).__init__()

        # ================================================================
        # Configuration
        # ================================================================
        self.size = size.lower()
        self.det_token_num = det_token_num
        self.num_keypoints = num_keypoints
        self.num_frames = num_frames
        self.decoder_layers = decoder_layers
        self.roi_output_size = roi_output_size
        self.masking_prob = 0.0
        self._last_roi_stats = {}
        
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
        
        logger.info(f'Building SIA-POSE-SIMPLE-DEC-ROI: size={self.size}, det_tokens={det_token_num}, '
                   f'num_frames={num_frames}, keypoints={num_keypoints}, roi_size={roi_output_size}x{roi_output_size}')

        # Print architecture diagram
        logger.info("Architecture Flow:\n" +
            "  Video Input [B, C, T, H, W]\n" +
            "    ↓\n" +
            "  Vision Encoder (ViT-{:s}/{:d}) - {:d} layers, {:d} heads, dim={:d}\n".format(
                self.size.upper(), self.patch_size, self.layers, self.heads, self.width) +
            "    ↓\n" +
            "  [Spatial Features] + [DET={:d}] Tokens\n".format(det_token_num) +
            "    ├─→ Detection Head (bboxes, human scores, class logits)\n" +
            "    └─→ ROI Extraction (roi_align {:d}x{:d}={:d} tokens per detection)\n".format(
                roi_output_size, roi_output_size, roi_output_size*roi_output_size) +
            "         └─→ Pose Decoder ({:d} layers, cross-attn to ROI features)\n".format(decoder_layers) +
            "              ├─→ Self-Attention among pose tokens\n" +
            "              ├─→ Cross-Attention to per-detection ROI features\n" +
            "              └─→ Keypoint Head ({:d} keypoints XY + visibility)\n".format(num_keypoints) +
            "  Output: pred_boxes, human_logits, pred_keypoints [B, N, K, 3]")

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
        
        # ================================================================
        # Module 3: ROI Positional Embedding
        # ================================================================
        if roi_output_size > 0:
            self.roi_pos_embed = nn.Parameter(torch.zeros(roi_output_size ** 2, self.width))
            nn.init.normal_(self.roi_pos_embed, std=0.02)

        # ================================================================
        # Module 4: ROI Pose Decoder (cross-attends only to ROI features)
        # ================================================================
        self.pose_decoder = nn.ModuleList([
            ROIPoseDecoderLayer(
                d_model=self.width,
                nhead=self.heads,
                dim_feedforward=self.width * 4,
                dropout=self.dropout_rate,
            )
            for _ in range(decoder_layers)
        ])
        self.pose_decoder_ln = nn.LayerNorm(self.width)

        # ================================================================
        # Module 5: Detection Heads (for bounding boxes)
        # ================================================================
        self.det_head_human = self.vision_encoder.human_embed
        self.det_head_bbox = self.vision_encoder.bbox_embed
        self.det_head_class = self.vision_encoder.proj

        # ================================================================
        # Module 6: Keypoint Heads (for pose estimation)
        # ================================================================
        # Learnable per-keypoint embeddings (efficient alternative to heavy projection)
        # Each keypoint type (nose, shoulder, ankle, etc.) has its own learned representation
        self.keypoint_embed = nn.Parameter(torch.zeros(num_keypoints, self.width))
        nn.init.normal_(self.keypoint_embed, std=0.02)

        self.keypoint_xy_head = MLP(self.width, self.width, 2, 3)       # Relative offsets [-1, 1]
        self.keypoint_vis_head = MLP(self.width, self.width // 2, 1, 2) # Visibility score

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
        return {'pose_token', 'keypoint_embed', 'roi_pos_embed',
                'vision_encoder.positional_embedding',
                'vision_encoder.class_embedding',
                'vision_encoder.temporal_positional_embedding'}

    def forward(self, video, masking_prob=None):
        """
        Forward pass through the ROI-based pose estimation architecture.
        
        Args:
            video: Input video tensor [B, C, T, H, W]
            masking_prob: Optional masking probability for augmentation
            
        Returns:
            Dictionary with detection and pose predictions, including aux outputs
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
        
        spatial_features = encoder_out['spatial_features']    # [B, H*W*T, D]
        det_x = encoder_out['det_tokens']                      # [B, det_num, D]
        pose_x = encoder_out['pose_tokens']                    # [B, pose_num, D]
        H_patches = encoder_out['H_patches']
        W_patches = encoder_out['W_patches']
        T = encoder_out['num_frames']

        outputs = {}

        # ================================================================
        # DETECTION HEAD STAGE: Predict bounding boxes and classes
        # ================================================================
        if self.det_head_class is not None:
            class_logits = det_x @ self.det_head_class
        else:
            class_logits = det_x
            
        bboxes = self.det_head_bbox(det_x).sigmoid()  # [B, det_num, 4]
        human_scores = self.det_head_human(det_x)
        
        outputs['pred_logits'] = class_logits
        outputs['pred_boxes'] = bboxes
        outputs['human_logits'] = human_scores

        # ================================================================
        # ROI EXTRACTION STAGE: Extract fixed-size ROI patches
        # ================================================================
        if self.roi_output_size > 0:
            roi_features, roi_mask, roi_stats = extract_roi_features_aligned(
                spatial_features=spatial_features,
                bboxes=bboxes.detach(),
                H_patches=H_patches,
                W_patches=W_patches,
                num_frames=T,
                output_size=self.roi_output_size,
                padding=0.1,
            )
            # Add learnable 2D positional embedding to ROI features
            roi_features = roi_features + self.roi_pos_embed.unsqueeze(0).unsqueeze(0)
            self._last_roi_stats = roi_stats
        else:
            roi_features = spatial_features
            roi_mask = None

        # ================================================================
        # POSE DECODER STAGE: Cross-attention to ROI features
        # ================================================================
        roi_mask_arg = roi_mask if (roi_mask is not None and roi_mask.any()) else None
        
        # Collect auxiliary outputs for intermediate supervision
        aux_outputs = []
        for decoder_layer in self.pose_decoder:
            pose_x = decoder_layer(pose_x, roi_features, roi_mask_arg)
            aux_outputs.append(self._predict_keypoints(
                self.pose_decoder_ln(pose_x), bboxes
            ))

        # Use final output
        final_kp_out = aux_outputs.pop()
        outputs['pred_keypoints'] = final_kp_out['pred_keypoints']

        # Attach intermediate outputs for auxiliary losses
        if len(aux_outputs) > 0:
            outputs['aux_outputs'] = [
                {**out_i, 'pred_logits': outputs['pred_logits'],
                 'pred_boxes': outputs['pred_boxes'], 'human_logits': outputs['human_logits']}
                for out_i in aux_outputs
            ]

        return outputs


    def _predict_keypoints(self, pose_x, bboxes):
        """
        Predict keypoints as bbox-relative offsets, then convert to global coordinates.

        Uses learnable per-keypoint embeddings conditioned on the parent detection's
        pose features. This is more parameter-efficient and gives each keypoint type
        (nose, shoulder, ankle, etc.) its own learned representation.

        Args:
            pose_x: Refined pose features [B, num_queries, D]
            bboxes: Predicted bboxes [B, num_queries, 4] in (cx, cy, w, h) format

        Returns:
            Dictionary with pred_keypoints [B, num_queries, num_keypoints, 3]
        """
        B_kp, num_det, D = pose_x.shape

        # Expand learnable keypoint embeddings: [num_keypoints, D] → [B, num_det, num_keypoints, D]
        # Each of the 17 keypoints (nose, shoulders, etc.) has its own learned embedding
        kp_embed = self.keypoint_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, num_keypoints, D]
        kp_embed = kp_embed.expand(B_kp, num_det, -1, -1)        # [B, num_det, num_keypoints, D]

        # Condition each keypoint on its parent detection's pose features
        # pose_x.unsqueeze(2): [B, num_det, 1, D] broadcasts to [B, num_det, num_keypoints, D]
        kp_features = kp_embed + pose_x.unsqueeze(2)  # [B, num_det, num_keypoints, D]

        kp_flat = kp_features.reshape(B_kp * num_det * self.num_keypoints, D)

        # Predict relative offsets in [-1, 1] range (shared MLP for all keypoints)
        keypoints_offset = self.keypoint_xy_head(kp_flat).tanh()
        # Predict visibility (shared MLP for all keypoints)
        keypoints_vis = self.keypoint_vis_head(kp_flat).sigmoid()

        keypoints_offset = keypoints_offset.reshape(B_kp, num_det, self.num_keypoints, 2)
        keypoints_vis = keypoints_vis.reshape(B_kp, num_det, self.num_keypoints, 1)

        # Convert relative offsets to global coordinates: kp = bbox_center + offset * bbox_size
        bbox_center = bboxes[:, :, :2].unsqueeze(2)  # [B, N, 1, 2] (cx, cy)
        bbox_size = bboxes[:, :, 2:].unsqueeze(2)    # [B, N, 1, 2] (w, h)
        keypoints_xy = (bbox_center + keypoints_offset * bbox_size).clamp(0, 1)

        pred_keypoints = torch.cat([keypoints_xy, keypoints_vis], dim=-1)
        return {'pred_keypoints': pred_keypoints}


    def _forward_encoder(self, video, masking_prob=0.0):
        """
        Extract features from the vision encoder.
        
        The encoder processes [CLS + spatial_patches] combined with
        [detection tokens] and [pose tokens].
        
        Returns:
            Dictionary with spatial_features, det_tokens, pose_tokens, and image dims
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
            
            # Pose tokens (using same positional embedding as detection for alignment)
            pose_tokens = self.pose_token + self.vision_encoder.det_positional_embedding
            pose_tokens = pose_tokens.unsqueeze(0).expand(B, -1, -1)
            
            # Concatenate: spatial_patches + pose_tokens + det_tokens
            x = torch.cat((x, pose_tokens, det_tokens), dim=1)
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
            pose_x = self.vision_encoder.dropout(
                x[:, -self.det_token_num-num_pose_tokens:-self.det_token_num]
            )
            det_x = self.vision_encoder.dropout(x[:, -self.det_token_num:])
        else:
            spatial_features = x
            pose_x = None
            det_x = None

        return {
            'spatial_features': spatial_features,
            'det_tokens': det_x,
            'pose_tokens': pose_x,
            'H_patches': H,
            'W_patches': W,
            'num_frames': T,
        }

