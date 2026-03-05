"""Embedding interpolation utilities for temporal and positional embeddings."""

import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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
