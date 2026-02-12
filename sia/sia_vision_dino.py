"""
Reusable DINOv2 Backbone Wrapper.

"""
import os
import sys
import logging

import torch
from torch import nn
import torch.nn.functional as F

from .sia_vision_clip import MLP

logger = logging.getLogger(__name__)


class DINOv2Backbone(nn.Module):
    """Reusable DINOv2 backbone wrapper.

    Loads a DINOv2 model via torch.hub and provides methods for:
    1. get_patch_features(x) — full forward, returns patch tokens only [B, N, D]
    2. prepare_tokens(x) — returns assembled token sequence [B, 1+R+N, D]
    3. forward_blocks(x) — runs tokens through transformer blocks
    4. forward_norm(x) — applies final LayerNorm
    """

    CONFIGS = {
        'b': {
            'model_name': 'dinov2_vitb14_reg',
            'embed_dim': 768,
            'num_heads': 12,
            'num_registers': 4,
        },
        'l': {
            'model_name': 'dinov2_vitl14_reg',
            'embed_dim': 1024,
            'num_heads': 16,
            'num_registers': 4,
        },
    }

    def __init__(self, size='b'):
        super().__init__()

        if size not in self.CONFIGS:
            raise ValueError(f"Unknown size '{size}'. Use 'b' or 'l'.")

        config = self.CONFIGS[size]
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.num_registers = config['num_registers']
        self.patch_size = 14

        # Ensure the dinov2 package is importable from hub cache
        hub_dir = torch.hub.get_dir()
        dinov2_repo = os.path.join(hub_dir, 'facebookresearch_dinov2_main')
        if os.path.isdir(dinov2_repo) and dinov2_repo not in sys.path:
            sys.path.insert(0, dinov2_repo)

        self.model = torch.hub.load(
            'facebookresearch/dinov2', config['model_name']
        )

    def pad_to_patch_size(self, x):
        """Pad H, W to nearest multiple of patch_size (14).

        Args:
            x: [B, C, H, W]
        Returns:
            Padded tensor (right and bottom zero-padding if needed)
        """
        _, _, H, W = x.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def get_patch_features(self, x):
        """Full DINOv2 forward pass, returns only patch tokens.

        Used by LED decoder models that need spatial features for cross-attention.

        Args:
            x: [B, C, H, W]
        Returns:
            patch_tokens: [B, num_patches, embed_dim]
        """
        x = self.pad_to_patch_size(x)
        dino_out = self.model.forward_features(x)
        if isinstance(dino_out, dict) and 'x_norm_patchtokens' in dino_out:
            return dino_out['x_norm_patchtokens']
        # Fallback: strip CLS + register tokens manually
        if isinstance(dino_out, dict):
            dino_out = dino_out.get('x_prenorm', dino_out.get('last_hidden_state'))
        return dino_out[:, 1 + self.num_registers:, :]

    def prepare_tokens(self, x):
        """Prepare DINOv2 token sequence (CLS + registers + patches with pos encoding).

        Used by encoder-only models that need to inject custom tokens before
        running through transformer blocks.

        Args:
            x: [B, C, H, W]
        Returns:
            tokens: [B, 1 + num_registers + num_patches, embed_dim]
        """
        x = self.pad_to_patch_size(x)
        return self.model.prepare_tokens_with_masks(x, masks=None)

    def forward_blocks(self, x):
        """Run token sequence through all DINOv2 transformer blocks.

        Args:
            x: [B, seq_len, embed_dim]
        Returns:
            x: [B, seq_len, embed_dim]
        """
        for blk in self.model.blocks:
            x = blk(x)
        return x

    def forward_norm(self, x):
        """Apply DINOv2's final LayerNorm.

        Args:
            x: [B, seq_len, embed_dim]
        Returns:
            x: [B, seq_len, embed_dim]
        """
        return self.model.norm(x)
