import os
import logging

import torch
from torch import nn
import torch.nn.functional as F

from .sia_vision_clip import clip_joint_l14, clip_joint_b16

logger = logging.getLogger(__name__)


class SIA_POSE(nn.Module):
    def __init__(self,
                 size='l',
                 pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
                 det_token_num=100,
                 num_frames=9,
                 # Pose detection parameters
                 num_keypoints=17,
                 pose_decoder_layers=2,
                 enable_pose=True):
        super(SIA_POSE, self).__init__()

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
        self.vision_encoder_drop_path_rate = 0 #0.1
        self.vision_encoder_checkpoint_num = 24
        self.is_pretrain = pretrain
        self.vision_width = 1024
        self.embed_dim = 768
        self.masking_prob = 0 #0.9

        # Extra techniques for vision encoder
        self.det_token_num = det_token_num
        if self.det_token_num > 0:
            print('Type: [DET] regression')
        else:
            print('Type: [PATCH] regression')

        # Pose detection parameters
        self.num_keypoints = num_keypoints
        self.pose_decoder_layers = pose_decoder_layers
        self.enable_pose = enable_pose
        if self.enable_pose:
            print(f'Pose detection enabled: {num_keypoints} keypoints, {pose_decoder_layers} decoder layers')

        # create modules.
        self.vision_encoder = self.build_vision_encoder()

        if pretrain:
            logger.info(f"Load pretrained weights from {pretrain}")
            state_dict = torch.load(pretrain, map_location='cpu', weights_only=False)['model']
            state_dict = interpolate_pos_embed_vit(state_dict, self) # interpolate temporal embeddings
            self.load_state_dict(state_dict, strict=False)

        if size.lower() == 'l':
            self.embed_dim = 768
        elif size.lower() == 'b':
            self.embed_dim = 512
        else:
            raise NotImplementedError(f"Size {size} not implemented")

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        return ret

    def forward(self, video):
        """Forward pass for bounding box regression.

        Args:
            video (torch.Tensor): The input images. Shape: [B,T,C,H,W].

        Returns:
            dict: Vision encoder outputs containing detection tokens projected to bounding boxes.

        """
        return self.encode_vision(video)
        

    def encode_vision(self, image, test=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,T,L,C].
            - pooled_vision_embeds (torch.Tensor): The pooled features. Shape: [B,T,C].

        """
        if image.ndim == 5:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
        else:
            image = image.unsqueeze(2)

        if not test and self.masking_prob > 0.0:
            return self.vision_encoder(
                image, masking_prob=self.masking_prob
            )

        return self.vision_encoder(image)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        encoder_name = self.vision_encoder_name
        if encoder_name == "vit_l14":
            vision_encoder = clip_joint_l14(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
                det_token_num=self.det_token_num,
                # Pose detection parameters
                num_keypoints=self.num_keypoints,
                pose_decoder_layers=self.pose_decoder_layers,
                enable_pose=self.enable_pose,
            )
        elif encoder_name == "vit_b16":
            vision_encoder = clip_joint_b16(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
                det_token_num=self.det_token_num,
                # Pose detection parameters
                num_keypoints=self.num_keypoints,
                pose_decoder_layers=self.pose_decoder_layers,
                enable_pose=self.enable_pose,
            )
        else:
            raise NotImplementedError(f"Not implemented: {encoder_name}")

        return vision_encoder


def interpolate_temporal_pos_embed(temp_embed_old, num_frames_new):
    """
    temp_embed_old: (1, num_frames_old, 1, d)
    Returns:
        temp_embed_new: (1, num_frames_new, 1, d)
    """
    temp_embed_old = temp_embed_old.squeeze(2).permute(
        0, 2, 1
    )  # (1, d, num_frames_old)
    temp_embed_new = F.interpolate(
        temp_embed_old, num_frames_new, mode="linear"
    )  # (1, d, num_frames_new)
    temp_embed_new = temp_embed_new.permute(0, 2, 1).unsqueeze(
        2
    )  # (1, num_frames_new, 1, d)
    return temp_embed_new

def load_temp_embed_with_mismatch(temp_embed_old, temp_embed_new, add_zero=False):
    """
    Add/Remove extra temporal_embeddings as needed.
    https://arxiv.org/abs/2104.00650 shows adding zero paddings works.

    temp_embed_old: (1, num_frames_old, 1, d)
    temp_embed_new: (1, num_frames_new, 1, d)
    add_zero: bool, if True, add zero, else, interpolate trained embeddings.
    """
    # TODO zero pad
    num_frms_new = temp_embed_new.shape[1]
    num_frms_old = temp_embed_old.shape[1]
    logger.info(f"Load temporal_embeddings, lengths: {num_frms_old}-->{num_frms_new}")
    if num_frms_new > num_frms_old:
        if add_zero:
            temp_embed_new[
                :, :num_frms_old
            ] = temp_embed_old  # untrained embeddings are zeros.
        else:
            temp_embed_new = interpolate_temporal_pos_embed(temp_embed_old, num_frms_new)
    elif num_frms_new < num_frms_old:
        temp_embed_new = temp_embed_old[:, :num_frms_new]
    else:  # =
        temp_embed_new = temp_embed_old
    return temp_embed_new

def interpolate_pos_embed_vit(state_dict, new_model):
    key = "vision_encoder.temporal_positional_embedding"
    if key in state_dict:
        vision_temp_embed_new = new_model.state_dict()[key]
        vision_temp_embed_new = vision_temp_embed_new.unsqueeze(2)  # [1, n, d] -> [1, n, 1, d]
        vision_temp_embed_old = state_dict[key]
        vision_temp_embed_old = vision_temp_embed_old.unsqueeze(2)

        state_dict[key] = load_temp_embed_with_mismatch(
            vision_temp_embed_old, vision_temp_embed_new, add_zero=False
        ).squeeze(2)

    return state_dict