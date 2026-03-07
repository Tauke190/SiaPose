import os
import logging

import torch
from torch import nn
import torch.nn.functional as F

from ..modules.sia_vision_clip import (
    VisionTransformer, MLP, inflate_weight, load_state_dict,
    _MODELS, PoseDecoderLayer
)
from ..embedding_utils import interpolate_pos_embed_vit

logger = logging.getLogger(__name__)


# ============================================================================
# Model Architecture: SIA Pose Decoder (with Token Splitting)
# ============================================================================
#
#  Input: Video [B, T, C, H, W]
#    ↓
#  Vision Encoder (ViT-B/L with temporal processing)
#    │ Conv1 → Spatial patches [B, HWT, D]
#    │ Temporal embedding across frames
#    │ Unified Query Tokens [B, N, D] (processed jointly with patches)
#    ↓
#  ┌─────────────────────────────────────────────────────────────┐
#  │ Transformer Backbone (12 or 24 layers)                      │
#  │ Self-attention over [spatial patches + query tokens]        │
#  │ Output: spatial_features [B, HWT, D]                       │
#  │         query_x          [B, N, D]                         │
#  └─────────────────────────────────────────────────────────────┘
#    │
#    │  TOKEN ROUTING (post-encoder, detached bboxes)
#    │  query_x [B, N, D]
#    │       ├ (direct use)
#    ├──→ det_x [B, N, D]   ← detection: use query tokens directly
#    │       └ pose_proj (Linear D→4D → GELU → Linear 4D→D)
#    └──→ pose_x [B, N, D]  ← pose: specialized features via projection
#    │
#    │  Gradients: encoder   ← both losses (shared, benefits from both tasks)
#    │             pose_proj ← keypoint losses only
#    │
#    ├─→ DETECTION BRANCH (det_x)
#    │       ↓ det_head_bbox  (MLP → sigmoid)
#    │       pred_boxes [B, N, 4]  (cx, cy, w, h) normalized
#    │       ↓ det_head_human (MLP)
#    │       human_logits [B, N, 2]
#    │
#    └─→ POSE DECODER BRANCH (pose_x)
#          pose_x [B, N, D]
#            ↓ (×decoder_layers) Self-Attn + Cross-Attn(spatial_features) + FFN
#            ↓ LayerNorm
#          refined pose_x [B, N, D]
#            ↓ keypoint_proj (MLP D → K×D, reshape to [B, N, K, D])
#          per-keypoint features [B, N, K, D]
#            ├─→ keypoint_xy_head  (MLP → tanh)  → bbox-relative offsets [-1,1]
#            │     → convert to absolute: kp = bbox_center + offset × bbox_size
#            └─→ keypoint_vis_head (MLP → sigmoid) → visibility [0,1]
#            ↓
#          pred_keypoints [B, N, K, 3]  (x, y, visibility)
#
#  TOTAL OUTPUTS:
#  {
#    'pred_boxes':     [B, N, 4],
#    'human_logits':   [B, N, 2],
#    'pred_keypoints': [B, N, 17, 3],
#  }
#
# ============================================================================
# SIA Pose Model with Lightweight Pose Decoder
# ============================================================================

class SIA_POSE_SIMPLE_DEC(nn.Module):
    """
    SIA Pose Model with unified query tokens for detection and pose

    Architecture breakdown:
    1. Vision Encoder: ViT processes [patches + query_tokens]
    2. Detection Head: query_tokens → detection token -> bbox and human score predictions
    3. Pose Decoder: query_tokens -> pose_tokens refined via cross-attention to spatial features
    4. Keypoint Head: pose-refined query_tokens → keypoint coordinates and visibility

    Single query token set ensures perfect alignment between detections and pose predictions that share positional embeddings
    """
    
    def __init__(self,
                 size='l',
                 pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
                 det_token_num=20,
                 num_frames=9,
                 num_keypoints=17,
                 decoder_layers=3):
        super(SIA_POSE_SIMPLE_DEC, self).__init__()

        # ================================================================
        # Configuration
        # ================================================================
        self.size = size.lower()
        self.det_token_num = det_token_num
        self.num_keypoints = num_keypoints
        self.num_frames = num_frames
        self.decoder_layers = decoder_layers
        self.masking_prob = 0.0
        
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
        
        logger.info(f'Building SIA-POSE-SIMPLE-DEC: size={self.size}, det_tokens={det_token_num}, '
                   f'num_frames={num_frames}, keypoints={num_keypoints}, decoder_layers={decoder_layers}')

        # Print architecture diagram
        logger.info("Architecture Flow:\n" +
            "  Video Input [B, C, T, H, W]\n" +
            "    ↓\n" +
            "  Vision Encoder (ViT-{:s}/{:d}) - {:d} layers, {:d} heads, dim={:d}\n".format(
                self.size.upper(), self.patch_size, self.layers, self.heads, self.width) +
            "    ↓\n" +
            "  [Spatial Patches] + [Query Tokens={:d}] (unified for detection+pose)\n".format(det_token_num) +
            "    ├─→ Detection token -> Detection Head (bboxes, human scores)\n" +
            "    |--> Query tokens"
            "    └─→ Pose Decoder ({:d} layers, cross-attn to spatial patches)\n".format(decoder_layers) +
            "         ├─→ Self-Attention among pose tokens tokens\n" +
            "         ├─→ Cross-Attention to spatial patches\n" +
            "         └─→ Keypoint Head ({:d} keypoints XY + visibility)\n".format(num_keypoints) +
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
        # Module 2: Unified Query Tokens (for both detection and pose)
        # ================================================================
        # Single set of query tokens predicts both boxes and keypoints.
        # This eliminates alignment issues: token[i] → box[i] + keypoints[i]
        self.query_token_num = det_token_num
        
        # ================================================================
        # Module 3: Pose Decoder (processes pose tokens after encoder)
        # ================================================================
        self.pose_decoder = nn.ModuleList([
            PoseDecoderLayer(
                d_model=self.width,
                nhead=self.heads,
                dim_feedforward=self.width * 4,
                dropout=self.dropout_rate,
            )
            for _ in range(decoder_layers)
        ])
        self.pose_decoder_ln = nn.LayerNorm(self.width)

        # ================================================================
        # Module 4: Pose Specialization Projection (post-encoder)
        # ================================================================
        # Only the pose branch needs specialization via cross-attention.
        # Detection tokens already encode bbox information in the encoder,
        # so no additional projection is needed there.
        #
        #   query_x [B, N, D]
        #       ├── (direct)   → det_x = query_x → Detection Heads (bbox, human)
        #       └── pose_proj  → pose_x → Pose Decoder → Keypoint Heads
        #
        self.pose_proj = nn.Sequential(
            nn.Linear(self.width, self.width * 4),
            nn.GELU(),
            nn.Linear(self.width * 4, self.width),
        )

        # ================================================================
        # Module 5: Detection Heads (for bounding boxes)
        # ================================================================
        # These are inherited from vision_encoder, but we reference them explicitly
        self.det_head_human = self.vision_encoder.human_embed  # [det_tokens] → human scores
        self.det_head_bbox = self.vision_encoder.bbox_embed    # [det_tokens] → bboxes

        # ================================================================
        # Module 6: Keypoint Heads (for pose estimation)
        # ================================================================
        # Use 2-layer MLP to project pose features to per-keypoint embeddings
        # This creates expressive, nonlinear per-keypoint features instead of simple additive embeddings
        self.keypoint_proj = nn.Sequential(
            nn.Linear(self.width, self.width * 4),
            nn.GELU(),
            nn.Linear(self.width * 4, num_keypoints * self.width),
        )

        self.keypoint_xy_head = MLP(self.width, self.width, 2, 3)       # → [x, y] coordinates [D] -> [2]
        self.keypoint_vis_head = MLP(self.width, self.width // 2, 1, 2) # → visibility score   [D] -> [1]

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
        return {'keypoint_embed',
                'vision_encoder.positional_embedding', 'vision_encoder.class_embedding',
                'vision_encoder.temporal_positional_embedding'}

    def forward(self, video, masking_prob=None):
        """
        Forward pass through the complete pose estimation architecture.
        
        Args:
            video: Input video tensor [B, T, C, H, W] (from dataset after batching)
            masking_prob: Optional masking probability for augmentation
            
        Returns:
            Dictionary with detection and pose predictions
        """
        if masking_prob is None:
            masking_prob = self.masking_prob
            
        # Normalize video format
        if video.ndim == 5:
            video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W] format
        else:
            video = video.unsqueeze(2)  # Add time dimension if missing

        # ================================================================
        # ENCODER STAGE: Vision Transformer processes all tokens
        # ================================================================
        B = video.shape[0]
        
        # Run vision encoder to get spatial features and query tokens
        # Query tokens serve both detection and pose estimation
        encoder_out = self._forward_encoder(video, masking_prob)

        spatial_features = encoder_out['spatial_features']    # [B, H*W*T, D]
        query_x = encoder_out['query_tokens']                  # [B, query_num, D]

        outputs = {}

        # ================================================================
        # TASK-SPECIFIC ROUTING: Split query tokens for detection and pose
        # ================================================================
        det_x = query_x                   # [B, N, D] — use directly for detection
        pose_x = self.pose_proj(query_x)  # [B, N, D] — specialize for pose refinement

        # ================================================================
        # DETECTION HEAD STAGE: Predict bounding boxes and human scores
        # ================================================================
        # Bounding boxes
        bboxes = self.det_head_bbox(det_x).sigmoid()  # [B, query_num, 4]

        # Human scores
        human_scores = self.det_head_human(det_x)     # [B, query_num, 2]

        # Diagnostic: log prediction statistics
        if self.training and B == 1:  # Only log for first batch in training
            with torch.no_grad():
                bbox_mean = bboxes.mean().item()
                bbox_std = bboxes.std().item()
                human_scores_max = human_scores.max().item()
                logger.debug(f"[Batch {B}] bboxes: mean={bbox_mean:.4f}, std={bbox_std:.4f} | "
                           f"human_scores: max={human_scores_max:.4f}")

        outputs['pred_boxes'] = bboxes
        outputs['human_logits'] = human_scores

        # ================================================================
        # POSE DECODER STAGE: Cross-attention to spatial features
        # Query tokens are refined by attending to spatial patches
        # Collect intermediate outputs for auxiliary losses
        # ================================================================
        aux_pose_outputs = []
        # pose_x already projected above via pose_proj
        for decoder_layer in self.pose_decoder:
            pose_x = decoder_layer(pose_x, spatial_features)  # Self-attn + cross-attn + FFN
            aux_pose_outputs.append(pose_x)

        pose_x = self.pose_decoder_ln(pose_x)  # [B, query_num, D]

        # ================================================================
        # KEYPOINT HEAD STAGE: Predict keypoint coordinates and visibility
        # Pass bboxes to anchor keypoint predictions
        # ================================================================
        pred_keypoints = self._extract_keypoints(pose_x, bboxes, B)

        # Diagnostic: log keypoint statistics
        if self.training and B == 1:
            with torch.no_grad():
                kp_xy = pred_keypoints[..., :2]
                kp_vis = pred_keypoints[..., 2]
                kp_xy_mean = kp_xy.mean().item()
                kp_vis_mean = kp_vis.mean().item()
                logger.debug(f"[Keypoints] xy: mean={kp_xy_mean:.4f}, vis: mean={kp_vis_mean:.4f}")

        outputs['pred_keypoints'] = pred_keypoints

        # ================================================================
        # AUXILIARY LOSSES: Commented out — only final decoder layer supervised
        # ================================================================
        # aux_outputs = []
        # for aux_pose_x in aux_pose_outputs[:-1]:  # Skip last layer (already in main output)
        #     aux_pose_x = self.pose_decoder_ln(aux_pose_x)
        #     aux_pred_keypoints = self._extract_keypoints(aux_pose_x, bboxes, B)
        #     aux_dict = {
        #         'pred_boxes': outputs['pred_boxes'],
        #         'human_logits': outputs['human_logits'],
        #         'pred_keypoints': aux_pred_keypoints,
        #     }
        #     aux_outputs.append(aux_dict)
        # if aux_outputs:
        #     outputs['aux_outputs'] = aux_outputs

        return outputs

    def _extract_keypoints(self, pose_x, bboxes, B):
        """
        Extract keypoint coordinates and visibility from pose features.

        CRITICAL: Predicts keypoints as bbox-relative offsets (tanh → [-1,1]) then
        converts to global normalized coordinates: kp = bbox_center + offset * bbox_size.
        This anchoring to the detected person is essential for learning — it gives the
        model a massive learning shortcut compared to predicting absolute positions.

        Uses a 2-layer MLP to project pose features into expressive per-keypoint embeddings,
        enabling the model to differentiate between keypoint types.

        Args:
            pose_x: [B, num_queries, D] pose token features
            bboxes: [B, num_queries, 4] predicted bboxes (cx, cy, w, h) normalized [0,1]
            B: batch size

        Returns:
            pred_keypoints: [B, num_queries, num_keypoints, 3] with (x, y, visibility) in [0,1]
        """
        B_kp, num_queries, D = pose_x.shape

        # Project each pose query to per-keypoint features using MLP
        # pose_x: [B, num_queries, D] → [B, num_queries, num_keypoints*D]
        kp_features_flat = self.keypoint_proj(pose_x)  # [B, num_queries, num_keypoints*D]

        # Reshape to [B, num_queries, num_keypoints, D]
        kp_features = kp_features_flat.reshape(B_kp, num_queries, self.num_keypoints, D)

        # Flatten for MLP heads: [B*num_queries*num_keypoints, D]
        kp_flat = kp_features.view(B_kp * num_queries * self.num_keypoints, D)

        # Predict keypoint offsets relative to bbox center (tanh → [-1, 1])
        # This anchors predictions to the detected bbox, making learning much easier
        keypoints_offset = self.keypoint_xy_head(kp_flat).tanh()  # [B*N*K, 2] in [-1, 1]

        # Predict keypoint visibility (shared MLP applied per keypoint)
        keypoints_vis = self.keypoint_vis_head(kp_flat).sigmoid()  # [B*N*K, 1]

        # Reshape offsets back to [B, num_queries, num_keypoints, 2]
        keypoints_offset = keypoints_offset.view(B_kp, num_queries, self.num_keypoints, 2)
        keypoints_vis = keypoints_vis.view(B_kp, num_queries, self.num_keypoints, 1)

        # Convert bbox-relative offsets to global normalized coordinates
        # kp = bbox_center + offset * bbox_size, where offset ∈ [-1, 1]
        bbox_center = bboxes[:, :, :2].unsqueeze(2)  # [B, N, 1, 2] (cx, cy)
        bbox_size = bboxes[:, :, 2:].unsqueeze(2)    # [B, N, 1, 2] (w, h)
        keypoints_xy = (bbox_center.detach() + keypoints_offset * bbox_size.detach()).clamp(0, 1)

        pred_keypoints = torch.cat([keypoints_xy, keypoints_vis], dim=-1)

        return pred_keypoints

    def _forward_encoder(self, video, masking_prob=0.0):
        """
        Extract features from the vision encoder.
        
        The encoder processes [CLS + spatial_patches + temporal] combined with
        [detection tokens] and [pose tokens] to produce enriched features.
        
        Returns:
            Dictionary with spatial_features and token outputs
        """
        # Video preprocessing (same as original)
        x = self.vision_encoder.conv1(video)  # [B, width, T, H_p, W_p]
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
        if self.vision_encoder.positional_embedding.shape[0] != x.shape[1]:
            temp_pos_embed = self.vision_encoder.InterpolateInitPosEmbed(
                temp_pos_embed, img_size=(video.shape[-2], video.shape[-1])
            )
        x = x + temp_pos_embed

        # Temporal positional embedding
        x = x[:, 1:]  # Remove CLS temporarily
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
        # Add unified [DET] tokens before encoder
        # ================================================================
        # Single set of query tokens that will predict both boxes and keypoints.
        # No separate pose tokens — the pose decoder receives det_x directly after encoding.
        if self.det_token_num > 0:
            # Query tokens (unified for detection and pose)
            query_tokens = self.vision_encoder.det_token + self.vision_encoder.det_positional_embedding
            query_tokens = query_tokens.unsqueeze(0).expand(B, -1, -1)

            # Concatenate: spatial_patches + query_tokens
            x = torch.cat((x, query_tokens), dim=1)
        else:
            pass

        # Forward through transformer
        x = self.vision_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch, width]
        x = self.vision_encoder.transformer(x)
        x = self.vision_encoder.ln_post(x)
        x = x.permute(1, 0, 2)  # [batch, seq_len, width]

        # ================================================================
        # Split outputs into spatial features and query tokens
        # ================================================================
        if self.det_token_num > 0:
            spatial_features = x[:, :-self.det_token_num]  # [B, H*W*T, D] all spatial patches
            query_x = self.vision_encoder.dropout(x[:, -self.det_token_num:])  # [B, det_num, D]
        else:
            spatial_features = x
            query_x = None

        return {
            'spatial_features': spatial_features,
            'query_tokens': query_x,
        }