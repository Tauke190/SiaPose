# ============================================================================
# SIA Pose: ROI-based Decoder Variant for COCO Keypoint Estimation
# ============================================================================
#
# ARCHITECTURE OVERVIEW:
# ──────────────────────────────────────────────────────────────────────────
#
# 1. ENCODER (Vision Transformer)
#    ├─ Input: Image → patches + temporal frames
#    ├─ Spatial Positional Embedding: [CLS] + patch positions + temporal positions
#    ├─ Detection Tokens: learnable [DET] tokens (N tokens, concatenated with patches)
#    ├─ Transformer Blocks: 12 layers (ViT-B/16) / 24 layers (ViT-L/14)
#    ├─ Output: spatial_features [B, H*W*T, D], det_x [B, N, D]
#
# 2. DETECTION HEADS
#    ├─ Input: det_x [B, N, D]
#    ├─ Class Projection: det_x @ proj → logits (for Hungarian matching)
#    ├─ BBox Regression: linear → (cx, cy, w, h) normalized, then sigmoid
#    ├─ Human Classifier: linear → human/background logits
#    └─ Output: bboxes [B, N, 4], human_logits [B, N, 2]
#
# 3. ROI EXTRACTION (CUDA-optimized)
#    ├─ Input: spatial_features [B, H*W*T, D], bboxes [B, N, 4]
#    ├─ Method: torchvision.ops.roi_align with fixed output size (P×P)
#    │  ├─ Converts normalized (cx, cy, w, h) → patch-space (x1, y1, x2, y2)
#    │  ├─ Bilinear interpolation + CUDA kernel (highly optimized)
#    │  ├─ Fixed output: PxP tokens per detection (no variable padding)
#    │  └─ ROI Positional Embedding: learnable 2D grid [P², D]
#    ├─ Temporal Averaging: if multiple frames, average before extraction
#    └─ Output: roi_features [B, N, P², D], roi_mask=None (no padding waste)
#
# 4. POSE DECODER (Learnable, Non-Positional)
#    ├─ Pose Queries: learnable tokens + det_positional_embedding (shared identity)
#    │  └─ CRITICAL: pose_query[i] ~ det_slot[i] after Hungarian matching
#    ├─ Multi-Layer Decoding (3 layers default):
#    │  ├─ Self-Attention: all N queries interact globally
#    │  ├─ Cross-Attention: query[i] attends ONLY to roi_features[i]
#    │  │  └─ No padding mask (roi_mask=None) → enables PyTorch FlashAttention
#    │  ├─ FFN: [D → 4D → D] with GELU activation
#    │  └─ Layer Normalization: pre-norm (norm → sublayer)
#    ├─ Intermediate Supervision: auxiliary losses from each layer
#    └─ Output: pose_x [B, N, D] (refined pose embeddings)
#
# 5. KEYPOINT HEADS (Efficient Bottleneck Design)
#    ├─ Input: pose_x [B, N, D]
#    ├─ Projection: linear [D → 17*256] (per-keypoint hidden feature bottleneck)
#    ├─ Per-Keypoint Processing:
#    │  ├─ Query-specific feature: [B, N, 17, 256]
#    │  ├─ Prediction head: [256 → 3] predicts (delta_x, delta_y, visibility)
#    │  ├─ Bbox-initialized absolute coords: sigmoid(logit(bbox_center) + delta)
#    │  └─ Visibility Score: sigmoid(0, 1) confidence of visibility
#    └─ Output: keypoints [B, N, 17, 3] (x, y, vis) normalized to [0, 1]
#
# Keypoint Decoupling from Bbox:
#    Keypoints are initialized at bbox_center in logit space, but the network
#    can freely predict anywhere in [0,1]. This avoids hard-anchoring to an
#    imperfect bbox while still benefiting from bbox as a warm-start prior.
#
# KEY DESIGN PRINCIPLES:
# ──────────────────────────────────────────────────────────────────────────
#
# ROI-focused Attention:
#   Unlike full-image methods, each pose query attends ONLY to patches within
#   the detected bounding box (via roi_align). This drastically reduces
#   attention computation and focuses the model on relevant person regions.
#
# Shared Slot Identity:
#   Pose queries use det_positional_embedding (not separate embeddings) so
#   pose_slot[i] and det_slot[i] have consistent identity. This ensures
#   Hungarian matching properly supervises both detection and pose outputs.
#
# Fixed-Size ROI Features:
#   roi_align with fixed output_size (P×P, e.g., 14×14=196 tokens) eliminates
#   padding waste and enables efficient PyTorch FlashAttention via SDPA.
#   Variable-length extraction is available (max_roi_cap mode) for compatibility.
#
# Auxiliary Supervision:
#   Each pose decoder layer produces keypoint predictions, enabling supervision
#   at multiple depths. Intermediate layer losses encourage gradual refinement.
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

    return roi_features, {}


# ============================================================================
# ViTDet Simple FPN
# ============================================================================

class ViTDetNeck(nn.Module):
    """
    ViTDet-style simple Feature Pyramid Network (He et al., 2022).

    Strictly follows ViTDet Appendix A.2:
        P2: 1/4  resolution — deconv → LN → GeLU → deconv
        P3: 1/8  resolution — one 2×2 deconvolution
        P4: 1/16 resolution — identity (ViT final feature map)
        P5: 1/32 resolution — stride-2 2×2 max pooling

    Per-level processing: 1×1 conv → LN → GeLU → 3×3 conv
    No top-down merging — ViT global self-attention already propagates full context.
    """
    def __init__(self, in_channels):
        super().__init__()
        C = in_channels
        # Spatial upsampling for P2 (two chained deconvs) and P3 (one deconv)
        self.deconv_p3   = nn.ConvTranspose2d(C, C, kernel_size=2, stride=2)
        self.deconv_p2_a = nn.ConvTranspose2d(C, C, kernel_size=2, stride=2)
        self.deconv_p2_b = nn.ConvTranspose2d(C, C, kernel_size=2, stride=2)
        # LN between the two P2 deconvolutions (ViTDet Appendix A.2:
        # "the first deconvolution is followed by LayerNorm (LN) and GeLU")
        self.deconv_p2_ln = nn.LayerNorm(C)
        # ViTDet Appendix A.2: "1×1 convolution with LN to reduce dimension to 256
        # and then a 3×3 convolution also with LN"
        # Strictly follows ViTDet — FPN operates at 256 channels internally.
        self.fpn_dim = 256
        self.proj = nn.ModuleList([nn.Conv2d(C, self.fpn_dim, kernel_size=1) for _ in range(4)])
        self.proj_norm = nn.ModuleList([nn.LayerNorm(self.fpn_dim) for _ in range(4)])
        self.smooth = nn.ModuleList([nn.Conv2d(self.fpn_dim, self.fpn_dim, kernel_size=3, padding=1) for _ in range(4)])
        self.smooth_norm = nn.ModuleList([nn.LayerNorm(self.fpn_dim) for _ in range(4)])

    def forward(self, feat_map):
        """
        Args:
            feat_map: [B, D, H_patches, W_patches]  e.g. [B, 768, 30, 40]
        Returns:
            list [P2, P3, P4, P5] each [B, D, H_k, W_k]
            For ViT-B/16 @ 480×640:
                P2: [B, 768, 120, 160]
                P3: [B, 768,  60,  80]
                P4: [B, 768,  30,  40]
                P5: [B, 768,  15,  20]
        """
        x = feat_map

        # P2: deconv → LN → GeLU → deconv  (strictly per ViTDet Appendix A.2)
        p2 = self.deconv_p2_a(x)
        p2 = self.deconv_p2_ln(p2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        p2 = F.gelu(p2)
        p2 = self.deconv_p2_b(p2)

        raw = [
            p2,                                        # P2: ×4 resolution
            self.deconv_p3(x),                         # P3: ×2 resolution
            x,                                         # P4: identity
            F.max_pool2d(x, kernel_size=2, stride=2),  # P5: ×0.5 resolution
        ]

        # Per-level: 1×1 conv → LN → GeLU → 3×3 conv → LN  (ViTDet Appendix A.2)
        out = []
        for i, f in enumerate(raw):
            f = self.proj[i](f)
            f = self.proj_norm[i](f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            f = F.gelu(f)
            f = self.smooth[i](f)
            f = self.smooth_norm[i](f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            out.append(f)
        return out


def assign_fpn_level(bboxes_norm, H_patches, W_patches, patch_size=16):
    """
    Assign each detection to an FPN pyramid level using the standard FPN rule
    (Lin et al. 2017, also used in ViTDet He et al. 2022):

        k = clamp( floor(k0 + log2(sqrt(area_pixels) / 56)), k_min=2, k_max=5 )

    k0=4 means P4 (1/16 resolution) is the reference level for 56×56px objects.
    Returned values {2,3,4,5} map to pyramid list indices {0,1,2,3} via k-2.

    For 480×640 input:
        Person 32×18px  (~small)  → P2 (1/4 res,  8 feature cells tall)
        Person 64×36px  (~medium) → P3 (1/8 res,  8 feature cells tall)
        Person 128×72px (~normal) → P4 (1/16 res, 8 feature cells tall)
        Person 300×160px (~large) → P5 (1/32 res, ~9 feature cells tall)

    Args:
        bboxes_norm: [B, N, 4] normalized (cx, cy, w, h) in [0, 1]
        H_patches:   int, feature map height in patch units (e.g. 30)
        W_patches:   int, feature map width in patch units  (e.g. 40)
        patch_size:  int, ViT patch size in pixels (16 for ViT-B/16, 14 for ViT-L/14)
    Returns:
        levels: [B, N] int64, values in {2, 3, 4, 5}
    """
    bw_px = bboxes_norm[:, :, 2] * (W_patches * patch_size)
    bh_px = bboxes_norm[:, :, 3] * (H_patches * patch_size)
    area_px = (bw_px * bh_px).clamp(min=1e-6)
    # k0=4: standard ViTDet/FPN formula (Lin et al. 2017, He et al. 2022)
    # P4 (1/16 res) is the reference level for 56×56px objects.
    k = 4.0 + torch.log2(torch.sqrt(area_px) / 56.0)
    return k.floor().long().clamp(2, 5)  # [B, N] in {2,3,4,5}


def extract_roi_features_fpn(pyramid, bboxes, levels, H_patches, W_patches,
                              output_size=7, padding=0.1):
    """
    Extract fixed-size ROI features from a 4-level ViTDet feature pyramid.

    For each detection, roi_align is called against the pyramid level assigned
    by assign_fpn_level(). One roi_align call per level; results are scattered
    back into a [B, N, P², D] output tensor preserving original detection order.

    Bounding boxes are kept in P4 patch-space coordinates. spatial_scale then
    maps those coordinates to the correct feature-map resolution at each level:
        P2 (4× upsampled from P4): spatial_scale = 4.0
        P3 (2× upsampled from P4): spatial_scale = 2.0
        P4 (base, same as P4):     spatial_scale = 1.0
        P5 (2× downsampled):       spatial_scale = 0.5

    Args:
        pyramid:     list [P2, P3, P4, P5], each [B, D, H_k, W_k]
        bboxes:      [B, N, 4] predicted boxes (cx, cy, w, h) normalized
        levels:      [B, N] int64 from assign_fpn_level, values in {2,3,4,5}
        H_patches:   int, P4 spatial height (number of patch rows)
        W_patches:   int, P4 spatial width  (number of patch columns)
        output_size: int P — each ROI pooled to P×P tokens
        padding:     float, fractional bbox padding (same as aligned path)
    Returns:
        roi_features: [B, N, P*P, D]
    """
    B, N, _ = bboxes.shape
    D = pyramid[0].shape[1]
    P = output_size
    device = bboxes.device
    dtype = pyramid[0].dtype

    spatial_scales = {2: 4.0, 3: 2.0, 4: 1.0, 5: 0.5}

    # Convert normalized bboxes → P4 patch-space (x1, y1, x2, y2)
    # Same coordinate conversion as extract_roi_features_aligned
    cx = bboxes[:, :, 0]
    cy = bboxes[:, :, 1]
    bw = bboxes[:, :, 2] * (1 + 2 * padding)
    bh = bboxes[:, :, 3] * (1 + 2 * padding)
    x1 = ((cx - bw / 2) * W_patches).clamp(min=0)
    y1 = ((cy - bh / 2) * H_patches).clamp(min=0)
    x2 = ((cx + bw / 2) * W_patches).clamp(max=W_patches)
    y2 = ((cy + bh / 2) * H_patches).clamp(max=H_patches)
    boxes_ps = torch.stack([x1, y1, x2, y2], dim=-1)  # [B, N, 4] in patch-space

    roi_out = torch.zeros(B, N, P * P, D, device=device, dtype=torch.float32)

    for lvl_idx, lvl in enumerate([2, 3, 4, 5]):
        mask = (levels == lvl)
        if not mask.any():
            continue
        b_idx, n_idx = mask.nonzero(as_tuple=True)  # [M] each
        rois = torch.cat([
            b_idx.float().unsqueeze(1),
            boxes_ps[b_idx, n_idx].float(),
        ], dim=1)  # [M, 5]

        lvl_out = roi_align(
            pyramid[lvl_idx].float(),
            rois,
            output_size=(P, P),
            spatial_scale=spatial_scales[lvl],
            aligned=True,
        )  # [M, D, P, P]

        # Reshape [M, D, P, P] → [M, P*P, D] and scatter into output
        roi_out[b_idx, n_idx] = lvl_out.reshape(-1, D, P * P).permute(0, 2, 1)

    return roi_out.to(dtype)


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

    def forward(self, queries, roi_features):
        """
        Args:
            queries:      [B, N, D]  pose queries (one per detection)
            roi_features: [B, N, roi_len, D]  per-detection ROI spatial features

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

        cross_out = self.cross_attn(
            query=q_flat,
            key=kv_flat,
            value=kv_flat,
        )[0]  # [B*N, 1, D]

        queries = queries + cross_out.reshape(B, N, D)

        # --- FFN ---
        queries = queries + self.ffn(self.norm3(queries))
        return queries

# Vision Transformer with ROI Decoder
class VisionTransformerSimpleDecoderROI(VisionTransformer):
    """
    Vision Transformer with ROI-based pose decoder.

    Detection tokens go through the full ViT encoder.
    Pose queries are NOT in the encoder — they cross-attend only to
    ROI features (patches inside detected bboxes) via decoder layers.

    Architecture:
        Encoder (ViT): [image_patches + det_tokens] -> spatial_features + det_x
        Det Heads:     det_x -> bboxes, human scores
        ROI Extract:   spatial_features[bbox_region] -> roi_features [B, N, roi_len, D]
        Pose Decoder:  pose_queries + cross-attn(roi_features) -> pose_x
        Pose Heads:    pose_x -> keypoints [B, N, 17, 3]
    """
    def __init__(
        self, input_resolution, patch_size, width, layers, heads, output_dim=None,
        kernel_size=1, num_frames=9, drop_path=0, checkpoint_num=0, dropout=0.,
        temp_embed=True, det_token_num=100,
        num_keypoints=17, decoder_layers=3,
        roi_output_size=14,
        use_fpn=False,
    ):
        # Parent encoder with NO pose handling
        super().__init__(
            input_resolution=input_resolution,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            output_dim=output_dim,
            kernel_size=kernel_size,
            num_frames=num_frames,
            drop_path=drop_path,
            checkpoint_num=checkpoint_num,
            dropout=dropout,
            temp_embed=temp_embed,
            det_token_num=det_token_num,
        )

        self.num_keypoints = num_keypoints
        self.patch_size = patch_size
        self.roi_output_size = roi_output_size  # P where each ROI -> PxP tokens

        # Remove the CLIP projection head (self.proj from parent) — this model only
        # detects humans (no multi-class), so pred_logits is never used in the loss.
        # Setting to None removes it from model.parameters() → no optimizer state waste.
        self.proj = None

        # Learnable pose queries that SHARE positional embedding with detection tokens
        # This ensures pose_slot[i] and det_slot[i] have the same "slot identity"
        # so Hungarian matching supervision is consistent across both outputs
        self.pose_token_num = det_token_num
        self.pose_token = nn.Parameter(torch.zeros(self.pose_token_num, width))
        nn.init.normal_(self.pose_token, std=0.02)
        # NOTE: We use det_positional_embedding (from parent) instead of separate pose PE
        # This is the KEY fix - pose slot i must share identity with det slot i

        # Learnable 2D positional embedding for fixed-size ROI grid (non-FPN path only).
        # When use_fpn=True this param is never used, so we skip registering it
        # entirely — avoids a DDP unused-parameter scan every iteration.
        self.use_fpn = use_fpn
        if roi_output_size > 0 and not use_fpn:
            self.roi_pos_embed = nn.Parameter(
                torch.zeros(roi_output_size ** 2, width)
            )
            nn.init.normal_(self.roi_pos_embed, std=0.02)

        # ViTDet FPN neck (optional)
        if use_fpn:
            self.fpn_neck = ViTDetNeck(in_channels=width)
            # Project FPN features (256) back to model width (768) after roi_align.
            # FPN operates at 256 (ViTDet paper); decoder cross-attn needs D=width.
            self.fpn_roi_proj = nn.Linear(self.fpn_neck.fpn_dim, width)
            # One positional embedding per FPN level (P2/P3/P4/P5 → indices 0/1/2/3)
            # so the decoder can distinguish which scale each ROI came from
            self.fpn_roi_pos_embeds = nn.ParameterList([
                nn.Parameter(torch.zeros(roi_output_size ** 2, width))
                for _ in range(4)
            ])
            for p in self.fpn_roi_pos_embeds:
                nn.init.normal_(p, std=0.02)

        # ROI pose decoder: self-attn + cross-attn(ROI) + FFN per layer
        self.pose_decoder_module = nn.ModuleList([
            ROIPoseDecoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=width * 4,
                dropout=dropout,
            )
            for _ in range(decoder_layers)
        ])
        self.pose_decoder_ln = nn.LayerNorm(width)

        # Keypoint heads: bottleneck projection (256× smaller than before)
        # Projects decoder output [B, N, D] → [B, N, 17*256]
        # Then simple linear: [B*N*17, 256] → [B*N*17, 3]
        self.kp_hidden_dim = 256
        self.keypoint_proj = nn.Linear(width, num_keypoints * self.kp_hidden_dim)
        self.keypoint_head = nn.Linear(self.kp_hidden_dim, 3)  # predicts (x, y, vis)

    def forward(self, x, masking_prob=0.0):
        """Forward pass with ROI-based pose decoding."""
        _, _, _, in_H, in_W = x.shape
        x = self.conv1(x)
        B, C, T, H, W = x.shape # T is number of frames
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        # [CLS] token + spatial positional embedding
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)

        temp_pos_embed = self.positional_embedding.to(x.dtype).unsqueeze(0)
        if self.positional_embedding.shape[1] != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(temp_pos_embed, img_size=(in_H, in_W))
        else:
            temp_pos_embed = self.positional_embedding
        x = x + temp_pos_embed

        # Temporal positional embedding
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        from einops import rearrange
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if hasattr(self, 'temporal_positional_embedding'):
            if x.size(1) == 1:
                x = x + self.temporal_positional_embedding.mean(1)
            else:
                x = x + self.temporal_positional_embedding
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        if masking_prob > 0.0:
            x = self.mask_tokens(x, masking_prob)

        # [DET] tokens go into encoder (NO pose tokens)
        if self.det_token_num > 0:
            det_tokens = self.det_token + self.det_positional_embedding
            det_tokens = det_tokens.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat((x, det_tokens), dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # BND -> NBD

        x = self.transformer(x)

        x = self.ln_post(x)

        # Split spatial features and det tokens
        if self.det_token_num > 0:
            spatial_features = x[:-self.det_token_num].permute(1, 0, 2)  # [B, H*W*T, D]
            det_x = self.dropout(x[-self.det_token_num:]).permute(1, 0, 2)  # [B, N, D]
        else:
            spatial_features = x.permute(1, 0, 2)
            det_x = self.dropout(x.reshape(H*W, T, B, C).mean(1)).permute(1, 0, 2)

        # --- Detection heads ---
        import torch.utils.checkpoint as checkpoint
        bboxes = checkpoint.checkpoint(self.bbox_embed, det_x, use_reentrant=False).sigmoid()
        human_scores = checkpoint.checkpoint(self.human_embed, det_x, use_reentrant=False)

        out = {'pred_boxes': bboxes, 'human_logits': human_scores}

        # --- ROI extraction ---
        if self.use_fpn:
            # Build 2D feature map from temporal-averaged spatial tokens
            if T > 1:
                sf = spatial_features.reshape(B, H * W, T, C).mean(dim=2)  # [B, H*W, D]
            else:
                sf = spatial_features  # [B, H*W, D]
            feat_map = sf.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, D, H, W]

            # Build 4-level pyramid and route each detection to the right level
            pyramid = self.fpn_neck(feat_map)
            levels = assign_fpn_level(
                bboxes, H, W, patch_size=self.patch_size
            )  # [B, N] in {2,3,4,5}
            roi_features = extract_roi_features_fpn(
                pyramid=pyramid,
                bboxes=bboxes,
                levels=levels,
                H_patches=H,
                W_patches=W,
                output_size=self.roi_output_size,
                padding=0.25,  # larger padding: more robust to imperfect bbox predictions
            )
            # FPN outputs 256-dim features; project back to D=width for the decoder.
            roi_features = self.fpn_roi_proj(roi_features)  # [B, N, P*P, 256] → [B, N, P*P, D]
            # Apply level-specific positional embeddings: each detection gets the
            # embedding for its assigned FPN level (levels ∈ {2,3,4,5} → idx {0,1,2,3})
            pos_embed_stack = torch.stack(list(self.fpn_roi_pos_embeds), dim=0)  # [4, P*P, D]
            level_idx = (levels - 2).clamp(0, 3)  # [B, N] → {0,1,2,3}
            per_det_embed = pos_embed_stack[level_idx]  # [B, N, P*P, D]
            roi_features = roi_features + per_det_embed
            roi_stats = {}
        elif self.roi_output_size > 0:
            roi_features, roi_stats = extract_roi_features_aligned(
                spatial_features=spatial_features,
                bboxes=bboxes,
                H_patches=H,
                W_patches=W,
                num_frames=T,
                output_size=self.roi_output_size,
                padding=0.1,
            )
            # Add learnable 2D positional embedding to ROI features
            # roi_features: [B, N, P*P, D], roi_pos_embed: [P*P, D]
            roi_features = roi_features + self.roi_pos_embed.unsqueeze(0).unsqueeze(0)

        self._last_roi_stats = roi_stats

        # --- ROI Pose decoder ---
        # Use det_positional_embedding (NOT separate pose PE) to share slot identity
        # This ensures pose_slot[i] corresponds to det_slot[i] after Hungarian matching
        pose_queries = self.pose_token + self.det_positional_embedding  # [N, D]
        pose_queries = pose_queries.unsqueeze(0).repeat(B, 1, 1)  # [B, N, D]

        # Run decoder layers — no intermediate supervision
        for layer in self.pose_decoder_module:
            pose_queries = layer(pose_queries, roi_features)

        # Apply LayerNorm once to the final decoder output
        out['pred_keypoints'] = self._predict_keypoints(
            self.pose_decoder_ln(pose_queries), bboxes
        )['pred_keypoints']

        return out

    def _predict_keypoints(self, pose_x, bboxes):
        """
        Predict keypoints using bbox as a soft prior initialization, not a hard anchor.

        Instead of: kp = bbox_center + offset * bbox_size  (fully bbox-anchored)
        We use:      kp = sigmoid(bbox_center + learned_delta)  (bbox-initialized but free)

        This decouples the final keypoint position from bbox accuracy:
        - The bbox center still provides a good initialization (faster convergence)
        - But the network can predict keypoints outside the bbox if needed
        - A bad bbox shifts the initialization, but the network can correct it
        """
        B_kp, num_det, D = pose_x.shape

        # Project to per-keypoint hidden features: [B, N, D] → [B, N, 17*256]
        kp_proj = self.keypoint_proj(pose_x)  # [B, N, 17*kp_hidden_dim]
        kp_features = kp_proj.reshape(B_kp, num_det, self.num_keypoints, self.kp_hidden_dim)

        # Flatten and predict (x, y, vis) for each keypoint
        kp_flat = kp_features.reshape(B_kp * num_det * self.num_keypoints, self.kp_hidden_dim)
        kp_out = self.keypoint_head(kp_flat)  # [B*N*17, 3]
        kp_out = kp_out.reshape(B_kp, num_det, self.num_keypoints, 3)

        keypoints_vis = kp_out[..., 2:3].sigmoid()

        # Bbox-initialized absolute prediction:
        # Use bbox center as reference point in logit space, add learned delta.
        # sigmoid(logit(cx) + delta_x) keeps prediction near bbox center at init
        # but allows free movement across the full image unlike tanh*bbox_size.
        bbox_center = bboxes[:, :, :2].unsqueeze(2)  # [B, N, 1, 2], values in (0,1)
        # Convert bbox center to logit space as initialization anchor
        bbox_center_clamped = bbox_center.clamp(1e-4, 1 - 1e-4)
        bbox_logit = torch.log(bbox_center_clamped / (1 - bbox_center_clamped))
        # Add predicted delta in logit space, then sigmoid back to [0,1]
        keypoints_xy = (bbox_logit + kp_out[..., :2]).sigmoid()

        pred_keypoints = torch.cat([keypoints_xy, keypoints_vis], dim=-1)
        return {'pred_keypoints': pred_keypoints}


# ============================================================================
# Model constructors
# ============================================================================

def clip_joint_b16_simple_dec_roi(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=100,
    dropout=0., num_keypoints=17, decoder_layers=3,
    roi_output_size=14, use_fpn=False,
):
    """ViT-B/16 with ROI pose decoder."""
    model = VisionTransformerSimpleDecoderROI(
        input_resolution=input_resolution, patch_size=16,
        width=768, layers=12, heads=12, output_dim=512,
        kernel_size=kernel_size, num_frames=num_frames,
        checkpoint_num=checkpoint_num,
        drop_path=drop_path, det_token_num=det_token_num,
        num_keypoints=num_keypoints, decoder_layers=decoder_layers,
        roi_output_size=roi_output_size,
        use_fpn=use_fpn,
    )
    if pretrained:
        model_name = pretrained if isinstance(pretrained, str) else "ViT-B/16"
        logger.info('load pretrained weights')
        state_dict = torch.load(_MODELS[model_name], map_location='cpu', weights_only=False)
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=16, center=center)
    return model.eval()


def clip_joint_l14_simple_dec_roi(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=20,
    dropout=0., num_keypoints=17, decoder_layers=3,
    roi_output_size=14, use_fpn=False,
):
    """ViT-L/14 with ROI pose decoder."""
    model = VisionTransformerSimpleDecoderROI(
        input_resolution=input_resolution, patch_size=14,
        width=1024, layers=24, heads=16, output_dim=768,
        kernel_size=kernel_size, num_frames=num_frames,
        checkpoint_num=checkpoint_num,
        drop_path=drop_path, det_token_num=det_token_num,
        num_keypoints=num_keypoints, decoder_layers=decoder_layers,
        roi_output_size=roi_output_size,
        use_fpn=use_fpn,
    )
    if pretrained:
        model_name = pretrained if isinstance(pretrained, str) else "ViT-L/14"
        logger.info('load pretrained weights')
        state_dict = torch.load(_MODELS[model_name], map_location='cpu', weights_only=False)
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=14, center=center)
    return model.eval()

# ============================================================================
# Utility functions (same as sia_pose_simple.py)
# ============================================================================

def interpolate_temporal_pos_embed(temp_embed_old, num_frames_new):
    """Interpolate temporal positional embeddings."""
    temp_embed_old = temp_embed_old.squeeze(2).permute(0, 2, 1)
    temp_embed_new = F.interpolate(temp_embed_old, num_frames_new, mode="linear")
    temp_embed_new = temp_embed_new.permute(0, 2, 1).unsqueeze(2)
    return temp_embed_new

def load_temp_embed_with_mismatch(temp_embed_old, temp_embed_new, add_zero=False):
    """Handle temporal embedding length mismatch."""
    num_frms_new = temp_embed_new.shape[1]
    num_frms_old = temp_embed_old.shape[1]
    logger.info(f"Load temporal_embeddings, lengths: {num_frms_old}-->{num_frms_new}")
    if num_frms_new > num_frms_old:
        if add_zero:
            temp_embed_new[:, :num_frms_old] = temp_embed_old
        else:
            temp_embed_new = interpolate_temporal_pos_embed(temp_embed_old, num_frms_new)
    elif num_frms_new < num_frms_old:
        temp_embed_new = temp_embed_old[:, :num_frms_new]
    else:
        temp_embed_new = temp_embed_old
    return temp_embed_new

def interpolate_pos_embed_vit(state_dict, new_model):
    """Interpolate positional embeddings for ViT."""
    key = "vision_encoder.temporal_positional_embedding"
    if key in state_dict:
        vision_temp_embed_new = new_model.state_dict()[key]
        vision_temp_embed_new = vision_temp_embed_new.unsqueeze(2)
        vision_temp_embed_old = state_dict[key]
        vision_temp_embed_old = vision_temp_embed_old.unsqueeze(2)

        state_dict[key] = load_temp_embed_with_mismatch(
            vision_temp_embed_old, vision_temp_embed_new, add_zero=False
        ).squeeze(2)

    return state_dict



class SIA_POSE_SIMPLE_DEC_ROI_BEST(nn.Module):
    """
    SIA Pose Model with ROI-based pose decoder.

    Detection tokens stay in the ViT encoder → predict bboxes.
    Pose queries are processed by a decoder that cross-attends only to
    ROI spatial features (patches inside each detected bbox).

    This focuses the pose decoder on the relevant person region instead
    of all ~1200 spatial patches.
    """
    def __init__(self,
                 size='l',
                 pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
                 det_token_num=20,
                 num_frames=1,
                 num_keypoints=17,
                 decoder_layers=3,
                 roi_output_size=14,
                 use_fpn=False):
        super(SIA_POSE_SIMPLE_DEC_ROI_BEST, self).__init__()

        if size.lower() == 'l':
            self.vision_encoder_name = 'vit_l14'
        elif size.lower() == 'b':
            self.vision_encoder_name = 'vit_b16'
        else:
            raise NotImplementedError(f"Size {size} not implemented")

        self.vision_encoder_pretrained = False
        self.inputs_image_res = 224
        self.vision_encoder_kernel_size = 1
        self.vision_encoder_center = True
        self.video_input_num_frames = num_frames
        self.vision_encoder_drop_path_rate = 0
        self.vision_encoder_checkpoint_num = 0
        self.is_pretrain = pretrain
        self.vision_width = 1024
        self.embed_dim = 768
        self.masking_prob = 0

        self.det_token_num = det_token_num
        self.num_keypoints = num_keypoints
        self.decoder_layers = decoder_layers
        self.roi_output_size = roi_output_size
        self.use_fpn = use_fpn

        if self.det_token_num > 0:
            print(f'Type: [DET] regression (DEC-ROI, {decoder_layers} decoder layers)')
        else:
            print(f'Type: [PATCH] regression (DEC-ROI, {decoder_layers} decoder layers)')

        if use_fpn:
            roi_info = f'ViTDet FPN {roi_output_size}x{roi_output_size}={roi_output_size**2} tokens/det (4 levels: P2-P5)'
        elif roi_output_size > 0:
            roi_info = f'roi_align {roi_output_size}x{roi_output_size}={roi_output_size**2} tokens'
        print(f'Dec-ROI pose: {num_keypoints} keypoints, {decoder_layers} decoder layers, {roi_info}')

        # Build vision encoder
        self.vision_encoder = self.build_vision_encoder()

        if pretrain:
            logger.info(f"Load pretrained weights from {pretrain}")
            state_dict = torch.load(pretrain, map_location='cpu', weights_only=False)['model']
            state_dict = interpolate_pos_embed_vit(state_dict, self)
            self.load_state_dict(state_dict, strict=False)

        if size.lower() == 'l':
            self.embed_dim = 768
        elif size.lower() == 'b':
            self.embed_dim = 512

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        return ret

    def forward(self, video):
        """Forward pass for pose estimation with ROI decoder."""
        return self.encode_vision(video)

    def encode_vision(self, image, test=False):
        """Encode image/videos as features."""
        if image.ndim == 5:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
        else:
            image = image.unsqueeze(2)

        if not test and self.masking_prob > 0.0:
            output = self.vision_encoder(image, masking_prob=self.masking_prob)
        else:
            output = self.vision_encoder(image)

        return output

    def build_vision_encoder(self):
        """Build vision encoder with ROI pose decoder."""
        encoder_name = self.vision_encoder_name
        if encoder_name == "vit_l14":
            vision_encoder = clip_joint_l14_simple_dec_roi(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
                det_token_num=self.det_token_num,
                num_keypoints=self.num_keypoints,
                decoder_layers=self.decoder_layers,
                roi_output_size=self.roi_output_size,
                use_fpn=self.use_fpn,
            )
        elif encoder_name == "vit_b16":
            vision_encoder = clip_joint_b16_simple_dec_roi(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
                det_token_num=self.det_token_num,
                num_keypoints=self.num_keypoints,
                decoder_layers=self.decoder_layers,
                roi_output_size=self.roi_output_size,
                use_fpn=self.use_fpn,
            )
        else:
            raise NotImplementedError(f"Not implemented: {encoder_name}")

        return vision_encoder
