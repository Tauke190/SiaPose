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

from .sia_vision import (
    VisionTransformer, MLP, inflate_weight, load_state_dict,
    _MODELS
)

logger = logging.getLogger(__name__)


class VisionTransformerSimple(VisionTransformer):
    """
    Vision Transformer without pose decoder.

    Keypoints are predicted directly from detection tokens without
    cross-attention refinement to spatial features.
    """
    def __init__(
        self, input_resolution, patch_size, width, layers, heads, output_dim=None,
        kernel_size=1, num_frames=9, drop_path=0, checkpoint_num=0, dropout=0.,
        temp_embed=True, det_token_num=100,
        num_keypoints=17,
    ):
        # Initialize parent WITHOUT pose decoder and WITHOUT lora
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
            lora=False,  # No LoRA in simple model
            num_keypoints=num_keypoints,
            pose_decoder_layers=0,  # No decoder layers
            enable_pose=False,  # Disable parent's pose handling
        )

        # Override: Add simple keypoint heads directly on detection tokens
        self.num_keypoints = num_keypoints
        self.enable_simple_pose = True

        # Dedicated pose tokens (randomly initialized, separate from det tokens)
        # These go through the encoder alongside det tokens but specialize for keypoints
        self.pose_token_num = det_token_num  # must match det_token_num for Hungarian matching
        self.pose_token = nn.Parameter(torch.zeros(self.pose_token_num, width))
        nn.init.normal_(self.pose_token, std=0.02)
        self.pose_positional_embedding = nn.Parameter(
            (width ** -0.5) * torch.randn(self.pose_token_num, width)
        )
        nn.init.normal_(self.pose_positional_embedding, std=0.02)

        # Simple keypoint embedding per pose token
        # Projects pose_token -> num_keypoints features
        self.keypoint_proj = nn.Linear(width, num_keypoints * width)

        # Keypoint prediction heads (same as original)
        self.simple_keypoint_xy_head = MLP(width, width, 2, 3)
        self.simple_keypoint_vis_head = MLP(width, width // 2, 1, 2)

    def forward(self, x, masking_prob=0.0):
        """Forward pass with simplified keypoint prediction."""
        _, _, _, in_H, in_W = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        # [CLS] Token and spatial pos
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)

        # Interpolate PE on-the-fly
        temp_pos_embed = self.positional_embedding.to(x.dtype).unsqueeze(0)
        if self.positional_embedding.shape[1] != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(temp_pos_embed, img_size=(in_H, in_W))
        else:
            temp_pos_embed = self.positional_embedding
        x = x + temp_pos_embed

        # Temporal pos
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

        # [DET] Tokens (for detection heads)
        if self.det_token_num > 0:
            det_tokens = self.det_token + self.det_positional_embedding
            det_tokens = det_tokens.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat((x, det_tokens), dim=1)

        # [POSE] Tokens (for keypoint regression, randomly initialized)
        pose_tokens = self.pose_token + self.pose_positional_embedding
        pose_tokens = pose_tokens.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat((x, pose_tokens), dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # BND -> NBD
        x = self.transformer(x)
        x = self.ln_post(x)

        # Split spatial, detection, and pose tokens
        # Sequence order: [spatial..., det_tokens..., pose_tokens...]
        # x shape: [N, B, D] where N = spatial + det_token_num + pose_token_num
        pose_x = self.dropout(x[-self.pose_token_num:]).permute(1, 0, 2)  # [B, pose_token_num, D]

        if self.det_token_num > 0:
            det_x = self.dropout(x[-(self.det_token_num + self.pose_token_num):-self.pose_token_num]).permute(1, 0, 2)  # [B, det_token_num, D]
        else:
            det_x = self.dropout(x[:-self.pose_token_num].reshape(H*W, T, B, C).mean(1)).permute(1, 0, 2)

        # Standard output heads (no LoRA)
        if self.proj is not None:
            class_scores = det_x @ self.proj
        else:
            class_scores = det_x

        import torch.utils.checkpoint as checkpoint
        bboxes = checkpoint.checkpoint(self.bbox_embed, det_x, use_reentrant=False).sigmoid()
        human_scores = checkpoint.checkpoint(self.human_embed, det_x, use_reentrant=False)

        out = {'pred_logits': class_scores, 'pred_boxes': bboxes, 'human_logits': human_scores}

        # Simple keypoint prediction (from dedicated pose tokens)
        if self.enable_simple_pose:
            B_kp, num_det, D = pose_x.shape

            # Project pose tokens to keypoint features
            # [B, num_det, D] -> [B, num_det, num_keypoints * D]
            kp_features = self.keypoint_proj(pose_x)
            # Reshape to [B, num_det, num_keypoints, D]
            kp_features = kp_features.view(B_kp, num_det, self.num_keypoints, D)

            # Flatten for MLP: [B * num_det * num_kp, D]
            kp_flat = kp_features.view(B_kp * num_det * self.num_keypoints, D)

            # Predict x, y coordinates (normalized 0-1)
            keypoints_xy = self.simple_keypoint_xy_head(kp_flat).sigmoid()
            # Predict visibility scores
            keypoints_vis = self.simple_keypoint_vis_head(kp_flat).sigmoid()

            # Reshape to [B, num_det, num_kp, 2] and [B, num_det, num_kp, 1]
            keypoints_xy = keypoints_xy.view(B_kp, num_det, self.num_keypoints, 2)
            keypoints_vis = keypoints_vis.view(B_kp, num_det, self.num_keypoints, 1)

            # Combine: [B, num_det, num_kp, 3] -> (x, y, visibility)
            pred_keypoints = torch.cat([keypoints_xy, keypoints_vis], dim=-1)
            out['pred_keypoints'] = pred_keypoints

        return out


def clip_joint_b16_simple(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=100,
    dropout=0., num_keypoints=17,
):
    """ViT-B/16 without pose decoder (no LoRA)."""
    model = VisionTransformerSimple(
        input_resolution=input_resolution, patch_size=16,
        width=768, layers=12, heads=12, output_dim=512,
        kernel_size=kernel_size, num_frames=num_frames,
        checkpoint_num=checkpoint_num,
        drop_path=drop_path, det_token_num=det_token_num,
        num_keypoints=num_keypoints,
    )
    if pretrained:
        model_name = pretrained if isinstance(pretrained, str) else "ViT-B/16"
        logger.info('load pretrained weights')
        state_dict = torch.load(_MODELS[model_name], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=16, center=center)
    return model.eval()


def clip_joint_l14_simple(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=100,
    dropout=0., num_keypoints=17,
):
    """ViT-L/14 without pose decoder (no LoRA)."""
    model = VisionTransformerSimple(
        input_resolution=input_resolution, patch_size=14,
        width=1024, layers=24, heads=16, output_dim=768,
        kernel_size=kernel_size, num_frames=num_frames,
        checkpoint_num=checkpoint_num,
        drop_path=drop_path, det_token_num=det_token_num,
        num_keypoints=num_keypoints,
    )
    if pretrained:
        model_name = pretrained if isinstance(pretrained, str) else "ViT-L/14"
        logger.info('load pretrained weights')
        state_dict = torch.load(_MODELS[model_name], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=14, center=center)
    return model.eval()


class SIA_POSE_SIMPLE(nn.Module):
    """
    Simplified SIA Pose Model without decoder.

    This model predicts keypoints directly from encoder detection tokens
    without cross-attention refinement to spatial features.

    Architecture:
    - ViT Encoder with detection tokens
    - Direct keypoint projection from detection tokens
    - No pose decoder (faster, simpler, but potentially less accurate)
    """
    def __init__(self,
                 size='l',
                 pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
                 det_token_num=100,
                 num_frames=9,
                 num_keypoints=17):
        super(SIA_POSE_SIMPLE, self).__init__()

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
        if self.det_token_num > 0:
            print('Type: [DET] regression (SIMPLE - no decoder)')
        else:
            print('Type: [PATCH] regression (SIMPLE - no decoder)')

        self.num_keypoints = num_keypoints
        print(f'Simple pose detection: {num_keypoints} keypoints, NO decoder')

        # Build vision encoder
        self.vision_encoder = self.build_vision_encoder()

        if pretrain:
            logger.info(f"Load pretrained weights from {pretrain}")
            state_dict = torch.load(pretrain, map_location='cpu')['model']
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
        """Forward pass for pose estimation without decoder."""
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
        """Build simplified vision encoder (no pose decoder)."""
        encoder_name = self.vision_encoder_name
        if encoder_name == "vit_l14":
            vision_encoder = clip_joint_l14_simple(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
                det_token_num=self.det_token_num,
                num_keypoints=self.num_keypoints,
            )
        elif encoder_name == "vit_b16":
            vision_encoder = clip_joint_b16_simple(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
                det_token_num=self.det_token_num,
                num_keypoints=self.num_keypoints,
            )
        else:
            raise NotImplementedError(f"Not implemented: {encoder_name}")

        return vision_encoder


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


