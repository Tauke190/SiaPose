# ============================================================================
# ROI Decoder variant for Pose Estimation
# ============================================================================
#
# Architecture:
#   Encoder (ViT): [image_patches + det_tokens] -> spatial_features + det_x
#   Det Heads:     det_x -> bboxes (cx, cy, w, h), human scores
#   ROI Extract:   spatial_features + bboxes -> per-detection ROI features
#   Pose Decoder:  pose_queries + cross-attn(ROI_features) -> pose_x
#   Pose Heads:    pose_x -> keypoints [B, N, 17, 3]
#
# Key difference from SIA_POSE_SIMPLE_DEC:
#   The decoder cross-attends to ROI features (only patches inside each
#   detected bbox) rather than ALL spatial patches. This focuses the
#   pose decoder on the relevant person region.
#
# ============================================================================

import os
import logging

import torch
from torch import nn
import torch.nn.functional as F

from .sia_vision_clip import (
    VisionTransformer, MLP, inflate_weight, load_state_dict,
    _MODELS, PoseDecoderLayer
)

logger = logging.getLogger(__name__)


# ============================================================================
# ROI feature extraction utilities
# ============================================================================

def extract_roi_features(spatial_features, bboxes, H_patches, W_patches,
                         num_frames=1, padding=0.1, min_patches=3):
    """
    Extract ROI features from spatial feature map using predicted bboxes.

    For each detection, selects the spatial patches that fall inside (or near)
    the predicted bounding box. Returns padded ROI features with a mask.

    Args:
        spatial_features: [B, H*W*T, D] encoder spatial output
        bboxes:          [B, N, 4] predicted boxes in (cx, cy, w, h) normalized format
        H_patches:       int, number of patch rows
        W_patches:       int, number of patch columns
        num_frames:      int, number of temporal frames
        padding:         float, fractional padding around bbox (0.1 = 10%)
        min_patches:     int, minimum patch grid size per dimension

    Returns:
        roi_features: [B, N, max_roi_len, D] padded ROI features per detection
        roi_mask:     [B, N, max_roi_len] bool mask (True = padding, to be ignored)
    """
    B, N, _ = bboxes.shape
    D = spatial_features.shape[-1]
    device = spatial_features.device

    # If multi-frame, average across temporal dimension first
    # spatial_features: [B, H*W*T, D] -> [B, H*W, D]
    if num_frames > 1:
        spatial_2d = spatial_features.reshape(B, H_patches * W_patches, num_frames, D).mean(dim=2)
    else:
        spatial_2d = spatial_features  # already [B, H*W, D]

    # Reshape to spatial grid: [B, H, W, D]
    spatial_grid = spatial_2d.reshape(B, H_patches, W_patches, D)

    # Convert (cx, cy, w, h) to patch-space (row_start, row_end, col_start, col_end)
    cx = bboxes[:, :, 0]  # [B, N]
    cy = bboxes[:, :, 1]
    bw = bboxes[:, :, 2]
    bh = bboxes[:, :, 3]

    # Add padding
    bw_padded = bw * (1 + 2 * padding)
    bh_padded = bh * (1 + 2 * padding)

    # Convert to patch indices (clamp to grid bounds)
    col_start = ((cx - bw_padded / 2) * W_patches).long().clamp(min=0)
    col_end   = ((cx + bw_padded / 2) * W_patches).long().clamp(max=W_patches)
    row_start = ((cy - bh_padded / 2) * H_patches).long().clamp(min=0)
    row_end   = ((cy + bh_padded / 2) * H_patches).long().clamp(max=H_patches)

    # Enforce minimum patch size
    col_mid = ((col_start + col_end) / 2).long()
    row_mid = ((row_start + row_end) / 2).long()
    half_min = min_patches // 2

    col_start = torch.minimum(col_start, (col_mid - half_min).clamp(min=0))
    col_end   = torch.maximum(col_end,   (col_mid + half_min + 1).clamp(max=W_patches))
    row_start = torch.minimum(row_start, (row_mid - half_min).clamp(min=0))
    row_end   = torch.maximum(row_end,   (row_mid + half_min + 1).clamp(max=H_patches))

    # Compute max ROI size for padding
    roi_heights = (row_end - row_start)  # [B, N]
    roi_widths  = (col_end - col_start)  # [B, N]
    roi_sizes   = roi_heights * roi_widths  # [B, N]
    max_roi_len = roi_sizes.max().item()

    if max_roi_len == 0:
        max_roi_len = min_patches * min_patches

    # Extract ROI features for each detection
    roi_features = torch.zeros(B, N, max_roi_len, D, device=device, dtype=spatial_features.dtype)
    roi_mask = torch.ones(B, N, max_roi_len, device=device, dtype=torch.bool)  # True = pad

    for b in range(B):
        for n in range(N):
            r0 = row_start[b, n].item()
            r1 = row_end[b, n].item()
            c0 = col_start[b, n].item()
            c1 = col_end[b, n].item()

            if r1 <= r0 or c1 <= c0:
                continue

            roi_patch = spatial_grid[b, r0:r1, c0:c1, :]  # [rh, rw, D]
            roi_len = (r1 - r0) * (c1 - c0)
            roi_flat = roi_patch.reshape(roi_len, D)

            roi_features[b, n, :roi_len] = roi_flat
            roi_mask[b, n, :roi_len] = False  # valid positions

    return roi_features, roi_mask


# ============================================================================
# ROI Pose Decoder Layer
# ============================================================================

class ROIPoseDecoderLayer(nn.Module):
    """
    Decoder layer where each pose query cross-attends to its own ROI features.

    Self-attention: all N pose queries interact with each other (global context).
    Cross-attention: each query_i attends only to roi_features_i (local ROI).
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
            roi_mask:     [B, N, roi_len] bool, True = padding (to be ignored)

        Returns:
            queries: [B, N, D] refined pose queries
        """
        B, N, D = queries.shape
        roi_len = roi_features.shape[2]

        # --- Self-attention (global, across all N queries) ---
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]

        # --- Cross-attention (per-detection, each query -> its ROI) ---
        # Reshape: treat each detection independently
        # queries:      [B, N, D]          -> [B*N, 1, D]
        # roi_features: [B, N, roi_len, D] -> [B*N, roi_len, D]
        # roi_mask:     [B, N, roi_len]    -> [B*N, roi_len]
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


# ============================================================================
# Vision Transformer with ROI Decoder
# ============================================================================

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
            num_keypoints=num_keypoints,
            pose_decoder_layers=0,
            enable_pose=False,
        )

        self.num_keypoints = num_keypoints
        self.patch_size = patch_size

        # Learnable pose queries that SHARE positional embedding with detection tokens
        # This ensures pose_slot[i] and det_slot[i] have the same "slot identity"
        # so Hungarian matching supervision is consistent across both outputs
        self.pose_token_num = det_token_num
        self.pose_token = nn.Parameter(torch.zeros(self.pose_token_num, width))
        nn.init.normal_(self.pose_token, std=0.02)
        # NOTE: We use det_positional_embedding (from parent) instead of separate pose PE
        # This is the KEY fix - pose slot i must share identity with det slot i

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

        # Keypoint heads
        self.keypoint_proj = nn.Linear(width, num_keypoints * width)
        self.simple_keypoint_xy_head = MLP(width, width, 2, 3)
        self.simple_keypoint_vis_head = MLP(width, width // 2, 1, 2)

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
        if self.proj is not None:
            class_scores = det_x @ self.proj
        else:
            class_scores = det_x

        import torch.utils.checkpoint as checkpoint
        bboxes = checkpoint.checkpoint(self.bbox_embed, det_x, use_reentrant=False).sigmoid()
        human_scores = checkpoint.checkpoint(self.human_embed, det_x, use_reentrant=False)

        out = {'pred_logits': class_scores, 'pred_boxes': bboxes, 'human_logits': human_scores}

        # --- ROI extraction ---
        # Use predicted bboxes (detached) to extract ROI features from spatial output
        roi_features, roi_mask = extract_roi_features(
            spatial_features=spatial_features,
            bboxes=bboxes.detach(),  # detach: don't backprop ROI selection through bbox head
            H_patches=H,
            W_patches=W,
            num_frames=T,
            padding=0.1,
            min_patches=3,
        )
        # roi_features: [B, N, max_roi_len, D]
        # roi_mask:     [B, N, max_roi_len]  (True = padding)

        # --- ROI Pose decoder ---
        # Use det_positional_embedding (NOT separate pose PE) to share slot identity
        # This ensures pose_slot[i] corresponds to det_slot[i] after Hungarian matching
        pose_queries = self.pose_token + self.det_positional_embedding  # [N, D]
        # Use repeat() not expand() to allocate new memory (required for proper backprop)
        pose_queries = pose_queries.unsqueeze(0).repeat(B, 1, 1)  # [B, N, D]

        for layer in self.pose_decoder_module:
            pose_queries = layer(pose_queries, roi_features, roi_mask)

        pose_x = self.pose_decoder_ln(pose_queries)  # [B, N, D]

        # --- Keypoint heads ---
        B_kp, num_det, D = pose_x.shape
        kp_features = self.keypoint_proj(pose_x)  # [B, N, 17*D]
        kp_features = kp_features.reshape(B_kp, num_det, self.num_keypoints, D)

        kp_flat = kp_features.reshape(B_kp * num_det * self.num_keypoints, D)
        keypoints_xy = self.simple_keypoint_xy_head(kp_flat).sigmoid()
        keypoints_vis = self.simple_keypoint_vis_head(kp_flat).sigmoid()

        keypoints_xy = keypoints_xy.reshape(B_kp, num_det, self.num_keypoints, 2)
        keypoints_vis = keypoints_vis.reshape(B_kp, num_det, self.num_keypoints, 1)
        pred_keypoints = torch.cat([keypoints_xy, keypoints_vis], dim=-1)
        out['pred_keypoints'] = pred_keypoints

        return out


# ============================================================================
# Model constructors
# ============================================================================

def clip_joint_b16_simple_dec_roi(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=100,
    dropout=0., num_keypoints=17, decoder_layers=3,
):
    """ViT-B/16 with ROI pose decoder."""
    model = VisionTransformerSimpleDecoderROI(
        input_resolution=input_resolution, patch_size=16,
        width=768, layers=12, heads=12, output_dim=512,
        kernel_size=kernel_size, num_frames=num_frames,
        checkpoint_num=checkpoint_num,
        drop_path=drop_path, det_token_num=det_token_num,
        num_keypoints=num_keypoints, decoder_layers=decoder_layers,
    )
    if pretrained:
        model_name = pretrained if isinstance(pretrained, str) else "ViT-B/16"
        logger.info('load pretrained weights')
        state_dict = torch.load(_MODELS[model_name], map_location='cpu', weights_only=False)
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=16, center=center)
    return model.eval()


def clip_joint_l14_simple_dec_roi(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=100,
    dropout=0., num_keypoints=17, decoder_layers=3,
):
    """ViT-L/14 with ROI pose decoder."""
    model = VisionTransformerSimpleDecoderROI(
        input_resolution=input_resolution, patch_size=14,
        width=1024, layers=24, heads=16, output_dim=768,
        kernel_size=kernel_size, num_frames=num_frames,
        checkpoint_num=checkpoint_num,
        drop_path=drop_path, det_token_num=det_token_num,
        num_keypoints=num_keypoints, decoder_layers=decoder_layers,
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


# ============================================================================
# Wrapper class
# ============================================================================

class SIA_POSE_SIMPLE_DEC_ROI(nn.Module):
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
                 det_token_num=100,
                 num_frames=9,
                 num_keypoints=17,
                 decoder_layers=3):
        super(SIA_POSE_SIMPLE_DEC_ROI, self).__init__()

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
        self.vision_encoder_checkpoint_num = 24
        self.is_pretrain = pretrain
        self.vision_width = 1024
        self.embed_dim = 768
        self.masking_prob = 0

        self.det_token_num = det_token_num
        self.num_keypoints = num_keypoints
        self.decoder_layers = decoder_layers

        if self.det_token_num > 0:
            print(f'Type: [DET] regression (DEC-ROI, {decoder_layers} decoder layers)')
        else:
            print(f'Type: [PATCH] regression (DEC-ROI, {decoder_layers} decoder layers)')

        print(f'Dec-ROI pose: {num_keypoints} keypoints, {decoder_layers} decoder layers, ROI cross-attn')

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
            return self.vision_encoder(image, masking_prob=self.masking_prob)

        return self.vision_encoder(image)

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
            )
        else:
            raise NotImplementedError(f"Not implemented: {encoder_name}")

        return vision_encoder
