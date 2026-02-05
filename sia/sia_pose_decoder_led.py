"""
SIA Pose Estimation Model - Late Encoder-Decoder (LED) Architecture.

In this architecture:
- The encoder processes ONLY spatial tokens (no detection tokens)
- Detection/pose queries are introduced in a separate decoder module
- The decoder cross-attends to encoder spatial features (like DETR)
- Output heads are attached to decoder outputs

This follows the DETR-style approach where queries don't participate in
encoder self-attention, which can be more efficient and allows the encoder
to focus purely on visual feature extraction.

The encoder uses the same ViCLIP architecture, just without detection tokens.
"""
import os
import logging

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import torch.utils.checkpoint as checkpoint

from .sia_vision import (
    MLP, PoseDecoderLayer, Transformer, inflate_weight, load_state_dict,
    _MODELS
)

logger = logging.getLogger(__name__)


class LEDDecoder(nn.Module):
    """Late Encoder-Decoder module.

    Takes learnable detection queries and refines them through cross-attention
    to encoder spatial features, then expands to keypoint queries.
    """
    def __init__(self, d_model=1024, nhead=8, num_layers=3, num_keypoints=17,
                 dim_feedforward=2048, dropout=0.1, det_token_num=100):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.det_token_num = det_token_num

        # Learnable detection queries (introduced here, not in encoder)
        self.det_queries = nn.Parameter(torch.zeros(det_token_num, d_model))
        nn.init.normal_(self.det_queries, std=0.02)

        # Learnable positional embeddings for detection queries
        self.det_pos_embed = nn.Parameter(torch.zeros(det_token_num, d_model))
        nn.init.normal_(self.det_pos_embed, std=0.02)

        # Learnable keypoint embeddings (one per joint type)
        self.keypoint_embed = nn.Parameter(torch.zeros(num_keypoints, d_model))
        nn.init.normal_(self.keypoint_embed, std=0.02)

        # Decoder layers for detection query refinement
        self.layers = nn.ModuleList([
            PoseDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Keypoint-specific decoder layer
        self.keypoint_layer = PoseDecoderLayer(d_model, nhead, dim_feedforward, dropout)

    def forward(self, spatial_features):
        """
        Args:
            spatial_features: [B, H*W*T, D] - encoder spatial output

        Returns:
            det_features: [B, det_token_num, D] - refined detection features
            keypoint_features: [B, det_token_num, num_keypoints, D] - per-detection keypoint features
        """
        B = spatial_features.shape[0]
        D = spatial_features.shape[-1]

        # Initialize detection queries with positional embeddings
        # [det_token_num, D] -> [B, det_token_num, D]
        det_queries = self.det_queries + self.det_pos_embed
        det_queries = det_queries.unsqueeze(0).expand(B, -1, -1)

        # Stage 1: Refine detection queries via cross-attention to spatial features
        for layer in self.layers:
            det_queries = layer(det_queries, spatial_features)

        # Stage 2: Expand to keypoint queries (one set of num_keypoints per detection)
        # keypoint_embed: [num_keypoints, D] -> [B, det_token_num, num_keypoints, D]
        kp_queries = self.keypoint_embed.unsqueeze(0).unsqueeze(0).expand(B, self.det_token_num, -1, -1)

        # Condition keypoints on their parent detection
        kp_queries = kp_queries + det_queries.unsqueeze(2)  # [B, det_token_num, num_keypoints, D]

        # Flatten for attention: [B, det_token_num*num_keypoints, D]
        kp_queries = kp_queries.contiguous().view(B, self.det_token_num * self.num_keypoints, D)

        # Cross-attend to spatial features for precise localization
        kp_queries = self.keypoint_layer(kp_queries, spatial_features)

        # Reshape back: [B, det_token_num, num_keypoints, D]
        keypoint_features = kp_queries.view(B, self.det_token_num, self.num_keypoints, D)

        return det_queries, keypoint_features


class VisionTransformerLED(nn.Module):
    """Vision Transformer with Late Encoder-Decoder architecture.

    The encoder processes only spatial tokens (patches + optional CLS).
    Detection queries are introduced in a separate decoder module that
    cross-attends to the encoder's spatial features.

    Uses the same ViCLIP transformer architecture, but without detection tokens
    in the encoder. Detection queries are introduced in the LED decoder.
    """
    def __init__(
        self, input_resolution, patch_size, width, layers, heads, output_dim=None,
        kernel_size=1, num_frames=9, drop_path=0, checkpoint_num=0, dropout=0.,
        temp_embed=True, det_token_num=100,
        num_keypoints=17, decoder_layers=3,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.width = width
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

        # Use the same ViCLIP transformer, but with det_token_num=0 (no detection tokens in encoder)
        # and lora=False (no LoRA adaptation needed since no det tokens)
        self.transformer = Transformer(
            width, layers, heads, drop_path=drop_path,
            checkpoint_num=checkpoint_num, dropout=dropout,
            det_token_num=0,  # No detection tokens in encoder
            lora=False,       # No LoRA needed
            num_frames=num_frames
        )

        self.ln_post = nn.LayerNorm(width)

        if output_dim is not None:
            self.proj = nn.Parameter(torch.empty(width, output_dim))
        else:
            self.proj = None

        self.dropout = nn.Dropout(dropout)
        self.det_token_num = det_token_num
        self.num_keypoints = num_keypoints

        # LED Decoder: introduces detection queries and cross-attends to spatial features
        self.led_decoder = LEDDecoder(
            d_model=width,
            nhead=heads,
            num_layers=decoder_layers,
            num_keypoints=num_keypoints,
            dim_feedforward=width * 4,
            dropout=dropout,
            det_token_num=det_token_num,
        )

        # Output heads (attached to decoder outputs)
        self.human_embed = MLP(width, width, 2, 3)
        self.bbox_embed = MLP(width, width, 4, 3)

        # Keypoint prediction heads
        self.keypoint_xy_head = MLP(width, width, 2, 3)
        self.keypoint_vis_head = MLP(width, width // 2, 1, 2)

    def get_num_layers(self):
        return len(self.transformer.resblocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'positional_embedding', 'class_embedding', 'temporal_positional_embedding'}

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        """Interpolate positional embeddings for different input sizes."""
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        patch_pos_embed = pos_embed[:, 1:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape

        P_H, P_W = self.input_resolution // self.patch_size, self.input_resolution // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False
        )
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
        return scale_pos_embed

    def forward(self, x, masking_prob=0.0):
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
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if hasattr(self, 'temporal_positional_embedding'):
            if x.size(1) == 1:
                x = x + self.temporal_positional_embedding.mean(1)
            else:
                x = x + self.temporal_positional_embedding
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        # NOTE: No detection tokens concatenated here! Clean encoder.

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # BND -> NBD
        x = self.transformer(x)
        x = self.ln_post(x)

        # Spatial features: [N, B, D] -> [B, H*W*T, D]
        spatial_features = x.permute(1, 0, 2)  # [B, H*W*T, D]

        # LED Decoder: cross-attend detection queries to spatial features
        det_features, keypoint_features = self.led_decoder(spatial_features)
        # det_features: [B, det_token_num, D]
        # keypoint_features: [B, det_token_num, num_keypoints, D]

        # Output heads
        det_features_dropped = self.dropout(det_features)

        if self.proj is not None:
            class_scores = det_features_dropped @ self.proj
        else:
            class_scores = det_features_dropped

        bboxes = checkpoint.checkpoint(self.bbox_embed, det_features_dropped, use_reentrant=False).sigmoid()
        human_scores = checkpoint.checkpoint(self.human_embed, det_features_dropped, use_reentrant=False)

        out = {'pred_logits': class_scores, 'pred_boxes': bboxes, 'human_logits': human_scores}

        # Keypoint predictions
        B_kp, num_det, num_kp, D_kp = keypoint_features.shape
        kp_flat = keypoint_features.view(B_kp * num_det * num_kp, D_kp)

        keypoints_xy = self.keypoint_xy_head(kp_flat).sigmoid()
        keypoints_vis = self.keypoint_vis_head(kp_flat).sigmoid()

        keypoints_xy = keypoints_xy.view(B_kp, num_det, num_kp, 2)
        keypoints_vis = keypoints_vis.view(B_kp, num_det, num_kp, 1)

        pred_keypoints = torch.cat([keypoints_xy, keypoints_vis], dim=-1)
        out['pred_keypoints'] = pred_keypoints

        return out


def clip_joint_b16_led(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=100,
    dropout=0., num_keypoints=17, decoder_layers=3,
):
    """ViT-B/16 with Late Encoder-Decoder architecture."""
    model = VisionTransformerLED(
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
        state_dict = torch.load(_MODELS[model_name], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=16, center=center)
    return model.eval()


def clip_joint_l14_led(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=100,
    dropout=0., num_keypoints=17, decoder_layers=3,
):
    """ViT-L/14 with Late Encoder-Decoder architecture."""
    model = VisionTransformerLED(
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
        state_dict = torch.load(_MODELS[model_name], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=14, center=center)
    return model.eval()


class SIA_POSE_DECODER_LED(nn.Module):
    """
    SIA Pose Model with Late Encoder-Decoder (LED) architecture.

    Architecture:
    - ViT Encoder: Processes only spatial tokens (patches), no detection tokens
    - LED Decoder: Introduces learnable detection queries, cross-attends to spatial features
    - Output Heads: BBox, Human classification, and Keypoint predictions

    This follows DETR-style design where detection queries don't participate
    in encoder self-attention, keeping encoder weights closer to pretrained CLIP.
    """
    def __init__(self,
                 size='l',
                 pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
                 det_token_num=100,
                 num_frames=9,
                 num_keypoints=17,
                 decoder_layers=3):
        super(SIA_POSE_DECODER_LED, self).__init__()

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

        print(f'Type: Late Encoder-Decoder (LED)')
        print(f'  - {det_token_num} detection queries in decoder')
        print(f'  - {decoder_layers} decoder layers')
        print(f'  - {num_keypoints} keypoints')

        # Build vision encoder with LED decoder
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
        """Forward pass for pose estimation with LED architecture."""
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
        """Build vision encoder with LED architecture."""
        encoder_name = self.vision_encoder_name
        if encoder_name == "vit_l14":
            vision_encoder = clip_joint_l14_led(
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
            vision_encoder = clip_joint_b16_led(
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
