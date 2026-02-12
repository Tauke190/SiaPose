"""
SIA Pose Heatmap Decoder - ViTPose Classic Decoder Style.

This model replaces MLP regression with a simple convolutional heatmap decoder.
Architecture: ViT Encoder -> spatial features -> 2x Deconv -> 1x1 Conv -> keypoint heatmaps

The decoder follows the ViTPose classic design:
    K = Conv1x1( Deconv( Deconv( Fout ) ) )

where Fout are the ViT spatial patch features.
"""
import os
import logging

import torch
from torch import nn
import torch.nn.functional as F

from .sia_vision_clip import (
    VisionTransformer, MLP, inflate_weight, load_state_dict,
    _MODELS
)

logger = logging.getLogger(__name__)


class ClassicDecoder(nn.Module):
    """ViTPose-style classic decoder with two deconvolution blocks."""

    def __init__(self, in_channels, num_keypoints=17, hidden_channels=256):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Two deconvolution blocks (each upsamples by 2x)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels, hidden_channels, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.deconv2 = nn.ConvTranspose2d(
            hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        # Final 1x1 conv to produce num_keypoints heatmaps
        self.final_conv = nn.Conv2d(hidden_channels, num_keypoints, kernel_size=1, padding=0)

    def forward(self, x):
        """
        Args:
            x: [B, D, H_p, W_p] spatial features at patch resolution

        Returns:
            heatmaps: [B, K, H_hm, W_hm] keypoint heatmaps (logits, not normalized)
        """
        # Deconv block 1: [B, D, H_p, W_p] -> [B, 256, H_p*2, W_p*2]
        x = F.relu(self.bn1(self.deconv1(x)))

        # Deconv block 2: [B, 256, H_p*2, W_p*2] -> [B, 256, H_p*4, W_p*4]
        x = F.relu(self.bn2(self.deconv2(x)))

        # Final conv: [B, 256, H_hm, W_hm] -> [B, K, H_hm, W_hm]
        return self.final_conv(x)


class VisionTransformerHeatmap(VisionTransformer):
    """
    Vision Transformer with ViTPose-style heatmap decoder.

    Keypoints are predicted as global heatmaps derived from spatial patch features,
    not from pose tokens. Pose tokens still go through the encoder for conditioning
    detection tokens, but their output is discarded.
    """

    def __init__(
        self, input_resolution, patch_size, width, layers, heads, output_dim=None,
        kernel_size=1, num_frames=9, drop_path=0, checkpoint_num=0, dropout=0.,
        temp_embed=True, det_token_num=100,
        num_keypoints=17,
    ):
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
            pose_decoder_layers=0,  # No decoder layers
            enable_pose=False,  # Disable parent's pose handling
        )

        self.num_keypoints = num_keypoints
        self.enable_heatmap_head = True

        # Dedicated pose tokens (randomly initialized, separate from det tokens)
        # These go through the encoder alongside det tokens but are NOT used for keypoints
        self.pose_token_num = det_token_num
        self.pose_token = nn.Parameter(torch.zeros(self.pose_token_num, width))
        nn.init.normal_(self.pose_token, std=0.02)
        self.pose_positional_embedding = nn.Parameter(
            (width ** -0.5) * torch.randn(self.pose_token_num, width)
        )
        nn.init.normal_(self.pose_positional_embedding, std=0.02)

        # Classic heatmap decoder (spatial features -> heatmaps)
        self.heatmap_decoder = ClassicDecoder(
            in_channels=width,
            num_keypoints=num_keypoints,
            hidden_channels=256
        )

    def forward(self, x, masking_prob=0.0):
        """Forward pass with heatmap keypoint prediction."""
        _, _, _, in_H, in_W = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        # Store spatial feature resolution for later
        H_p, W_p = H, W
        N_spatial = H_p * W_p * T

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

        # [POSE] Tokens (kept in encoder but NOT used for keypoint prediction)
        pose_tokens_embed = self.pose_token + self.pose_positional_embedding
        pose_tokens_embed = pose_tokens_embed.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat((x, pose_tokens_embed), dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # BND -> NBD
        x = self.transformer(x)
        x = self.ln_post(x)

        # Extract spatial features and det tokens
        # Sequence order: [spatial..., det_tokens..., pose_tokens...]
        # x shape: [N, B, D] where N = spatial + det_token_num + pose_token_num

        # Spatial features (excluding CLS which was stripped earlier)
        spatial_features = x[:N_spatial].permute(1, 0, 2)  # [B, N_spatial, D]

        # Detection tokens
        if self.det_token_num > 0:
            det_x = self.dropout(x[-(self.det_token_num + self.pose_token_num):-self.pose_token_num]).permute(1, 0, 2)  # [B, det_token_num, D]
        else:
            det_x = self.dropout(x[:-self.pose_token_num].reshape(H_p*W_p, T, B, C).mean(1)).permute(1, 0, 2)

        # Standard output heads (detection)
        if self.proj is not None:
            class_scores = det_x @ self.proj
        else:
            class_scores = det_x

        import torch.utils.checkpoint as checkpoint
        bboxes = checkpoint.checkpoint(self.bbox_embed, det_x, use_reentrant=False).sigmoid()
        human_scores = checkpoint.checkpoint(self.human_embed, det_x, use_reentrant=False)

        out = {'pred_logits': class_scores, 'pred_boxes': bboxes, 'human_logits': human_scores}

        # Heatmap keypoint prediction (from spatial features)
        if self.enable_heatmap_head:
            # Reshape spatial features to 2D: [B, N_spatial, D] -> [B, D, H_p, W_p]
            spatial_2d = spatial_features.view(B, H_p, W_p, T, -1)  # [B, H_p, W_p, T, D]
            spatial_2d = spatial_2d.permute(0, 4, 3, 1, 2).reshape(B, -1, H_p, W_p)  # [B, D*T, H_p, W_p]
            # For single frame (T=1), just take D
            if T == 1:
                spatial_2d = spatial_features.view(B, H_p, W_p, -1).permute(0, 3, 1, 2)  # [B, D, H_p, W_p]

            # Apply heatmap decoder
            heatmap_logits = self.heatmap_decoder(spatial_2d)  # [B, K, H_hm, W_hm]

            # Store both logits and softmax heatmaps
            out['pred_heatmap_logits'] = heatmap_logits
            out['pred_heatmaps'] = F.softmax(heatmap_logits.view(B, self.num_keypoints, -1), dim=-1).view_as(heatmap_logits)

            # For backward compatibility with HungarianMatcher and PostProcessPose:
            # Decode heatmaps to normalized coordinates via argmax
            H_hm, W_hm = heatmap_logits.shape[-2:]

            # Argmax decode: [B, K, H_hm, W_hm] -> [B, K, 2]
            heatmap_flat = heatmap_logits.view(B, self.num_keypoints, -1)
            indices = torch.argmax(heatmap_flat, dim=-1)  # [B, K]

            # Convert flat indices to (y, x) coordinates
            y_idx = indices // W_hm  # [B, K]
            x_idx = indices % W_hm   # [B, K]

            # Normalize to [0, 1]
            pred_x = x_idx.float() / max(W_hm - 1, 1)
            pred_y = y_idx.float() / max(H_hm - 1, 1)

            # Visibility: for heatmap model, use the max heatmap value as proxy for confidence
            max_vals = torch.amax(heatmap_flat, dim=-1)  # [B, K]

            # Stack: [B, K, 3] -> (x, y, visibility)
            pred_keypoints = torch.stack([pred_x, pred_y, max_vals], dim=-1)
            out['pred_keypoints'] = pred_keypoints

        return out


def clip_joint_b16_heatmap(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=100,
    dropout=0., num_keypoints=17,
):
    """ViT-B/16 with heatmap decoder"""
    model = VisionTransformerHeatmap(
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
        state_dict = torch.load(_MODELS[model_name], map_location='cpu', weights_only=False)
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=16, center=center)
    return model.eval()


def clip_joint_l14_heatmap(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=100,
    dropout=0., num_keypoints=17,
):
    """ViT-L/14 with heatmap decoder"""
    model = VisionTransformerHeatmap(
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
        state_dict = torch.load(_MODELS[model_name], map_location='cpu', weights_only=False)
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=14, center=center)
    return model.eval()


class SIA_POSE_HEATMAP(nn.Module):
    """
    SIA Pose Model with ViTPose-style heatmap decoder.

    This model predicts keypoints as spatial heatmaps using a simple convolutional
    decoder on top of ViT spatial features, following the ViTPose classic design.

    Architecture:
    - ViT Encoder with detection and pose tokens
    - Heatmap decoder on spatial features: Conv -> Deconv -> Deconv -> Conv
    - Global heatmap output: [B, 17, H_hm, W_hm]
    - Bounding box detection from det tokens (unchanged)
    """

    def __init__(self,
                 size='l',
                 pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
                 det_token_num=100,
                 num_frames=9,
                 num_keypoints=17):
        super(SIA_POSE_HEATMAP, self).__init__()

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
            print('Type: [DET] heatmap decoder (ViTPose-style)')
        else:
            print('Type: [PATCH] heatmap decoder (ViTPose-style)')

        self.num_keypoints = num_keypoints
        print(f'Heatmap pose detection: {num_keypoints} keypoints with classic decoder')

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
        """Forward pass for pose estimation with heatmap decoder."""
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
        """Build vision encoder with heatmap decoder."""
        encoder_name = self.vision_encoder_name
        if encoder_name == "vit_l14":
            vision_encoder = clip_joint_l14_heatmap(
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
            vision_encoder = clip_joint_b16_heatmap(
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