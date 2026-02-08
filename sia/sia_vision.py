#!/usr/bin/env python
import os
import logging
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
#from timm.models.registry import register_model

import torch.utils.checkpoint as checkpoint

import math

logger = logging.getLogger(__name__)

# On P1, model extracted from https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
MODEL_PATH = ''
_MODELS = {
    "ViT-L/14": os.path.join(MODEL_PATH, "ViCLIP-L_InternVid-FLT-10M.pth"),
    "ViT-B/16": os.path.join(MODEL_PATH, "ViCLIP-B-InternVid-FLT-10M.pth"),
}

# YOLOS: Added Custom MLP
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# ============================================================================
# Pose Decoder Module for Multi-Person Keypoint Detection
# ============================================================================

class PoseDecoderLayer(nn.Module):
    """Single decoder layer with self-attn + cross-attn + FFN"""
    def __init__(self, d_model=1024, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention: queries (Q) attend to spatial features (K, V)
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

    def forward(self, queries, spatial_features):
        """
        queries: [B, num_queries, D] - detection or keypoint queries
        spatial_features: [B, H*W*T, D] - encoder output (preserved, not discarded!)
        """
        # Self-attention: queries interact with each other
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]

        # Cross-attention: Q=queries, K=V=spatial_features
        q = self.norm2(queries)
        queries = queries + self.cross_attn(
            query=q,                    # What we want to find
            key=spatial_features,       # Where to look
            value=spatial_features      # What to extract
        )[0]

        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class PoseDecoder(nn.Module):
    """Full pose decoder with detection refinement + keypoint localization"""
    def __init__(self, d_model=1024, nhead=8, num_layers=2, num_keypoints=17, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Learnable keypoint embeddings (one per joint type)
        self.keypoint_embed = nn.Parameter(torch.zeros(num_keypoints, d_model))
        nn.init.normal_(self.keypoint_embed, std=0.02)

        # Decoder layers for detection refinement
        self.layers = nn.ModuleList([
            PoseDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])

        # Keypoint-specific layer (after expanding to per-detection keypoints)
        self.keypoint_layer = PoseDecoderLayer(d_model, nhead, dim_feedforward, dropout)

    def forward(self, det_queries, spatial_features):
        """
        det_queries: [B, num_det, D] - initial detection tokens from encoder
        spatial_features: [B, H*W*T, D] - encoder spatial output (NOT discarded)
        Returns:
            refined_det: [B, num_det, D] - refined detection features
            keypoint_features: [B, num_det, num_keypoints, D] - per-detection keypoint features
        """
        B, num_det, D = det_queries.shape

        # Stage 1: Refine detection queries via cross-attention to spatial features
        for layer in self.layers:
            det_queries = layer(det_queries, spatial_features)

        # Stage 2: Expand to keypoint queries (one set of num_keypoints per detection)
        # keypoint_embed: [num_keypoints, D] → [B, num_det, num_keypoints, D]
        kp_queries = self.keypoint_embed.unsqueeze(0).unsqueeze(0).expand(B, num_det, -1, -1)

        # Condition keypoints on their parent detection
        kp_queries = kp_queries + det_queries.unsqueeze(2)  # [B, num_det, num_keypoints, D]

        # Flatten for attention: [B, num_det*num_keypoints, D]
        # Use contiguous() + view() or reshape() to handle non-contiguous tensors
        kp_queries = kp_queries.contiguous().view(B, num_det * self.num_keypoints, D)

        # Cross-attend to spatial features for precise localization
        kp_queries = self.keypoint_layer(kp_queries, spatial_features)

        # Reshape back: [B, num_det, num_keypoints, D]
        keypoint_features = kp_queries.view(B, num_det, self.num_keypoints, D)

        return det_queries, keypoint_features


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
 

class TransformerMLP(nn.Module):
    def __init__(self, d_model, dropout=0.):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.drop1 = nn.Dropout(dropout)
        self.c_proj = nn.Linear(d_model * 4, d_model)
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.drop1(x)
        x = self.c_proj(x)
        x = self.drop2(x)
        return x
        
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., dropout=0., det_token_num=100):
        super().__init__()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # logger.info(f'Droppath: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
   
        self.mlp = TransformerMLP(d_model, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.det_token_num = det_token_num

    def attention(self, x):
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, x):
        x = x + self.drop_path1(self.attention(self.ln_1(x)))
        x = x + self.drop_path2(self.mlp(self.ln_2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, width, layers, heads, drop_path=0., checkpoint_num=0, dropout=0., det_token_num=100, num_frames=9):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        in_frame = num_frames
        for idx in range(layers):
            self.resblocks.append(ResidualAttentionBlock(width,
                                                         heads,
                                                         drop_path=dpr[idx],
                                                         dropout=dropout,
                                                         det_token_num=det_token_num))
        self.checkpoint_num = checkpoint_num

    def forward(self, x):
        for idx, blk in enumerate(self.resblocks):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self, input_resolution, patch_size, width, layers, heads, output_dim=None,
        kernel_size=1, num_frames=9, drop_path=0, checkpoint_num=0, dropout=0.,
        temp_embed=True, det_token_num=100,
        num_keypoints=17, pose_decoder_layers=2, enable_pose=True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv3d(
            3, width,
            (kernel_size, patch_size, patch_size),
            (kernel_size, patch_size, patch_size),
            (0, 0, 0), bias=False
        )

        self.input_resolution = input_resolution
        self.patch_size = patch_size

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        if temp_embed:
            self.temporal_positional_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(
            width, layers, heads, drop_path=drop_path, checkpoint_num=checkpoint_num,
            dropout=dropout, det_token_num=det_token_num, num_frames=num_frames)

        self.ln_post = nn.LayerNorm(width)
        if output_dim is not None:
            self.proj = nn.Parameter(torch.empty(width, output_dim))
        else:
            self.proj = None # Action logits MLP

        self.dropout = nn.Dropout(dropout)

        #YOLOS: Added MLP for Human logits, BBoxes and Action logits
        self.human_embed = MLP(width, width, 2, 3)
        self.bbox_embed = MLP(width, width, 4, 3)


        self.det_token_num = det_token_num
        # YOLOS: Added [DET] Tokens
        if self.det_token_num > 0:
            self.det_token = nn.Parameter(torch.zeros(self.det_token_num, width))
            nn.init.normal_(self.det_token, std=0.02)

            # YOLOS: Added PE for [DET] Tokens
            self.det_positional_embedding = nn.Parameter(scale * torch.randn(self.det_token_num, width))
            nn.init.normal_(self.det_positional_embedding, std=0.02)

        # ============================================================================
        # Pose Detection: Decoder + Keypoint Heads
        # ============================================================================
        self.enable_pose = enable_pose
        self.num_keypoints = num_keypoints

        if self.enable_pose and self.det_token_num > 0:
            # Pose decoder: cross-attends to spatial features
            self.pose_decoder = PoseDecoder(
                d_model=width,
                nhead=heads,
                num_layers=pose_decoder_layers,
                num_keypoints=num_keypoints,
                dim_feedforward=width * 4,
                dropout=dropout
            )

            # Keypoint prediction heads
            # x, y coordinates (normalized 0-1)
            self.keypoint_xy_head = MLP(width, width, 2, 3)
            # Visibility score (0-1)
            self.keypoint_vis_head = MLP(width, width // 2, 1, 2)

    def get_num_layers(self):
        return len(self.transformer.resblocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'positional_embedding', 'class_embedding', 'temporal_positional_embedding'}
    
    def mask_tokens(self, inputs, masking_prob=0.0):
        B, L, _ = inputs.shape

        # This is different from text as we are masking a fix number of tokens
        Lm = int(masking_prob * L)
        masked_indices = torch.zeros(B, L)
        indices = torch.argsort(torch.rand_like(masked_indices), dim=-1)[:, :Lm]
        batch_indices = (
            torch.arange(masked_indices.shape[0]).unsqueeze(-1).expand_as(indices)
        )
        masked_indices[batch_indices, indices] = 1

        masked_indices = masked_indices.bool()

        return inputs[~masked_indices].reshape(B, -1, inputs.shape[-1])

    # YOLOS: Added Online Interpolation for Positional Embedding 
    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:,None]
        patch_pos_embed = pos_embed[:, 1:, :]
        patch_pos_embed = patch_pos_embed.transpose(1,2)
        B, E, Q = patch_pos_embed.shape


        P_H, P_W = self.input_resolution // self.patch_size, self.input_resolution // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H//self.patch_size, W//self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
        return scale_pos_embed

    def forward(self, x, masking_prob=0.0):
        _, _, _, in_H, in_W = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        # [CLS] Token and spatial pos
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        # YOLOS: interpolate PE on-the-fly
        # interpolate init pe
        temp_pos_embed = self.positional_embedding.to(x.dtype).unsqueeze(0)
        if self.positional_embedding.shape[1] != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(temp_pos_embed, img_size=(in_H, in_W))
        else:
            temp_pos_embed = self.positional_embedding
        x = x + temp_pos_embed

        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if hasattr(self, 'temporal_positional_embedding'):
            if x.size(1) == 1:
                # This is a workaround for unused parameter issue
                x = x + self.temporal_positional_embedding.mean(1)
            else:
                x = x + self.temporal_positional_embedding
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        if masking_prob > 0.0:
            x = self.mask_tokens(x, masking_prob)

        # [DET] Tokens and corresponding pos
        # x = torch.cat((cls_tokens, x), dim=1) Removed [CLS] Tokens
        if self.det_token_num > 0:
            det_tokens = self.det_token + self.det_positional_embedding
            det_tokens = det_tokens + torch.zeros(B, det_tokens.shape[0], det_tokens.shape[1]).to(det_tokens.device)
            x = torch.cat((x, det_tokens), dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  #BND -> NBD
        x = self.transformer(x)

        x = self.ln_post(x)

        # ============================================================================
        # Split spatial features and detection tokens (PRESERVE spatial features)
        # ============================================================================
        if self.det_token_num > 0:
            # Spatial features: [N - det_token_num, B, D] -> [B, H*W*T, D]
            spatial_features = x[:-self.det_token_num].permute(1, 0, 2)  # [B, H*W*T, D]
            # Detection tokens: [det_token_num, B, D] -> [B, det_token_num, D]
            det_x = self.dropout(x[-self.det_token_num:]).permute(1, 0, 2)  # [B, det_token_num, D]
        else:
            spatial_features = x.permute(1, 0, 2)  # [B, H*W*T, D]
            det_x = self.dropout(x.reshape(H*W, T, B, C).mean(1)).permute(1, 0, 2)

        # ============================================================================
        # Pose Decoder: Cross-attend detection tokens to spatial features
        # ============================================================================
        if self.enable_pose and self.det_token_num > 0:
            # Use pose decoder to refine detection queries and extract keypoint features
            refined_det, keypoint_features = self.pose_decoder(det_x, spatial_features)
            # refined_det: [B, det_token_num, D]
            # keypoint_features: [B, det_token_num, num_keypoints, D]

            # Use refined detection features for bbox/human heads
            det_for_heads = refined_det
        else:
            det_for_heads = det_x
            keypoint_features = None

        # ============================================================================
        # Output Heads
        # ============================================================================
        if self.proj is not None:
            class_scores = det_for_heads @ self.proj
        else:
            class_scores = det_for_heads

        bboxes = checkpoint.checkpoint(self.bbox_embed, det_for_heads, use_reentrant=False).sigmoid()
        if self.det_token_num == 0:
            box_bias = F.pad(torch.stack(torch.meshgrid(torch.linspace(0,1,W), torch.linspace(0,1,H))).reshape(2, W*H).permute(1,0), (0,2), 'constant', 0)
            bboxes = bboxes + box_bias.to(bboxes.device)
        human_scores = checkpoint.checkpoint(self.human_embed, det_for_heads, use_reentrant=False)

        out = {'pred_logits': class_scores, 'pred_boxes': bboxes, 'human_logits': human_scores}

        # ============================================================================
        # Keypoint Predictions (if pose detection is enabled)
        # ============================================================================
        if self.enable_pose and keypoint_features is not None:
            B_kp, num_det, num_kp, D_kp = keypoint_features.shape
            # Flatten for MLP: [B * num_det * num_kp, D]
            kp_flat = keypoint_features.view(B_kp * num_det * num_kp, D_kp)

            # Predict x, y coordinates (normalized 0-1)
            keypoints_xy = self.keypoint_xy_head(kp_flat).sigmoid()  # [B*num_det*num_kp, 2]
            # Predict visibility scores
            keypoints_vis = self.keypoint_vis_head(kp_flat).sigmoid()  # [B*num_det*num_kp, 1]

            # Reshape to [B, num_det, num_kp, 2] and [B, num_det, num_kp, 1]
            keypoints_xy = keypoints_xy.view(B_kp, num_det, num_kp, 2)
            keypoints_vis = keypoints_vis.view(B_kp, num_det, num_kp, 1)

            # Combine: [B, num_det, num_kp, 3] -> (x, y, visibility)
            pred_keypoints = torch.cat([keypoints_xy, keypoints_vis], dim=-1)
            out['pred_keypoints'] = pred_keypoints

        return out

def inflate_weight(weight_2d, time_dim, center=True):
    logger.info(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d

def load_state_dict(model, state_dict, input_resolution=224, patch_size=16, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 2:
                logger.info(f'Ignore: {k}')
                continue
            logger.info(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)

    pos_embed_checkpoint = state_dict['positional_embedding']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = (input_resolution // patch_size) ** 2
    orig_size = int((pos_embed_checkpoint.shape[-2] - 1) ** 0.5)
    new_size = int(num_patches ** 0.5)
    if orig_size != new_size:
        logger.info(f'Pos_emb from {orig_size} to {new_size}')
        extra_tokens = pos_embed_checkpoint[:1]
        pos_tokens = pos_embed_checkpoint[1:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(0, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
        state_dict['positional_embedding'] = new_pos_embed
    
    message = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Load pretrained weights: {message}")

#@register_model
def clip_joint_b16(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=100,
    dropout=0.,
    # Pose detection parameters
    num_keypoints=17, pose_decoder_layers=2, enable_pose=True,
):
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=16,
        width=768, layers=12, heads=12, output_dim=512,
        kernel_size=kernel_size, num_frames=num_frames,
        checkpoint_num=checkpoint_num,
        drop_path=drop_path, det_token_num=det_token_num,
        # Pose detection
        num_keypoints=num_keypoints, pose_decoder_layers=pose_decoder_layers, enable_pose=enable_pose,
    )
    # raise NotImplementedError
    if pretrained:
        if isinstance(pretrained, str):
            model_name = pretrained
        else:
            model_name = "ViT-B/16"

        logger.info('load pretrained weights')
        state_dict = torch.load(_MODELS[model_name], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=16, center=center)
    return model.eval()

#@register_model
def clip_joint_l14(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=100,
    dropout=0.,
    # Pose detection parameters
    num_keypoints=17, pose_decoder_layers=2, enable_pose=True,
):
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=14,
        width=1024, layers=24, heads=16, output_dim=768,
        kernel_size=kernel_size, num_frames=num_frames,
        checkpoint_num=checkpoint_num,
        drop_path=drop_path, det_token_num=det_token_num,
        # Pose detection
        num_keypoints=num_keypoints, pose_decoder_layers=pose_decoder_layers, enable_pose=enable_pose,
    )

    if pretrained:
        if isinstance(pretrained, str):
            model_name = pretrained
        else:
            model_name = "ViT-L/14"
        logger.info('load pretrained weights')
        state_dict = torch.load(_MODELS[model_name], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=14, center=center)
    return model.eval()
