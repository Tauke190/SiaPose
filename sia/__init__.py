from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .sia_pose import SIA_POSE
from .sia_pose_simple import SIA_POSE_SIMPLE
from .sia_pose_heatmap import SIA_POSE_HEATMAP
from .sia_pose_dino_simple import SIA_POSE_DINO_SIMPLE
from torch import nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, sigmoid_focal_loss
import numpy as np
import cv2
import os
import torch

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from scipy.optimize import linear_sum_assignment

def get_sia_pose(size='l',
              pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
              det_token_num=100,
              num_frames=9,
              # Pose detection parameters
              num_keypoints=17,
              pose_decoder_layers=2,
              enable_pose=True):

    sia_model = SIA_POSE(size=size,
                        pretrain=pretrain,
                        det_token_num=det_token_num,
                        num_frames=num_frames,
                        # Pose detection parameters
                        num_keypoints=num_keypoints,
                        pose_decoder_layers=pose_decoder_layers,
                        enable_pose=enable_pose,
                        )
    m = {'sia':sia_model}

    return m


def get_sia_pose_simple(size='l',
                   pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
                   det_token_num=100,
                   num_frames=9,
                   num_keypoints=17):
    """
    Get simplified SIA pose model (no decoder).

    This model predicts keypoints directly from encoder detection tokens
    without cross-attention refinement to spatial features.
    """
    sia_model = SIA_POSE_SIMPLE(
        size=size,
        pretrain=pretrain,
        det_token_num=det_token_num,
        num_frames=num_frames,
        num_keypoints=num_keypoints,
    )
    m = {'sia': sia_model}
    return m


def get_sia_pose_heatmap(size='l',
                          pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
                          det_token_num=100,
                          num_frames=9,
                          num_keypoints=17):
    """
    Get SIA pose model with ViTPose-style heatmap decoder.

    Architecture:
    - ViT Encoder with detection and pose tokens
    - Classic deconv heatmap decoder on spatial features
    - Global heatmap output: K = Conv1x1( Deconv( Deconv( Fout ) ) )
    """
    sia_model = SIA_POSE_HEATMAP(
        size=size,
        pretrain=pretrain,
        det_token_num=det_token_num,
        num_frames=num_frames,
        num_keypoints=num_keypoints,
    )
    m = {'sia': sia_model}
    return m


def get_sia_pose_dino(size='b',
                      det_token_num=100,
                      num_frames=1,
                      num_keypoints=17,
                      decoder_layers=3):
    """
    Get SIA pose model with DINOv2 backbone + LED decoder.

    DINOv2 weights are loaded automatically via torch.hub (no manual pretrain path needed).

    Args:
        size: 'b' for ViT-B/14 (768d) or 'l' for ViT-L/14 (1024d)
        det_token_num: Number of detection queries
        num_frames: Number of input frames
        num_keypoints: Number of keypoints (17 for COCO)
        decoder_layers: Number of LED decoder layers
    """
    sia_model = SIA_POSE_DINO(
        size=size,
        det_token_num=det_token_num,
        num_frames=num_frames,
        num_keypoints=num_keypoints,
        decoder_layers=decoder_layers,
    )
    m = {'sia': sia_model}
    return m


def get_sia_pose_dino_simple(size='b',
                             det_token_num=100,
                             num_frames=1,
                             num_keypoints=17):
    """
    Get SIA pose model with DINOv2 backbone, encoder-only (no decoder).

    Detection and pose tokens go through DINOv2's encoder alongside patch tokens.
    Keypoints are predicted directly from pose tokens.

    Args:
        size: 'b' for ViT-B/14 (768d) or 'l' for ViT-L/14 (1024d)
        det_token_num: Number of detection/pose queries
        num_frames: Number of input frames
        num_keypoints: Number of keypoints (17 for COCO)
    """
    sia_model = SIA_POSE_DINO_SIMPLE(
        size=size,
        det_token_num=det_token_num,
        num_frames=num_frames,
        num_keypoints=num_keypoints,
    )
    m = {'sia': sia_model}
    return m


###################################
# For closed-set action detection #
###################################
# YOLOS: Added Hungarian Matcher
# REMOVED cost_class to avoid matching for actions
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_human: float = 1,
                 cost_keypoint: float = 0): #HUMAN + KEYPOINT
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            cost_keypoint: This is the relative weight of the keypoint L1 error in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class # remove in the future
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_human = cost_human #HUMAN
        self.cost_keypoint = cost_keypoint #KEYPOINT
        assert cost_bbox != 0 or cost_giou != 0 or cost_human != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                 "pred_keypoints": (optional) Tensor of dim [batch_size, num_queries, num_keypoints, 3] with keypoints

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "keypoints": (optional) Tensor of dim [num_target_boxes, num_keypoints, 3] containing keypoints

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_human = outputs["human_logits"].flatten(0, 1).softmax(-1) #HUMAN

        # Also concat the target labels and boxes
        #tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_human = torch.zeros(len(tgt_bbox)).int().to(tgt_bbox.device) #HUMAN

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.

        cost_human = -out_human[:, tgt_human] #HUMAN

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_human * cost_human #HUMAN

        # Add keypoint cost if available
        if self.cost_keypoint > 0 and 'pred_keypoints' in outputs and len(targets) > 0 and 'keypoints' in targets[0]:
            # out_keypoints: [bs * num_queries, num_keypoints, 3]
            out_keypoints = outputs["pred_keypoints"].flatten(0, 1)
            # tgt_keypoints: [num_targets, num_keypoints, 3]
            tgt_keypoints = torch.cat([v["keypoints"] for v in targets])

            # Only use x, y coordinates for matching (not visibility)
            out_kp_xy = out_keypoints[..., :2].flatten(1)  # [bs * num_queries, num_keypoints * 2]
            tgt_kp_xy = tgt_keypoints[..., :2].flatten(1)  # [num_targets, num_keypoints * 2]

            # L1 cost for keypoints
            cost_kp = torch.cdist(out_kp_xy, tgt_kp_xy, p=1)
            C = C + self.cost_keypoint * cost_kp

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

# YOLOS: Added Loss
# Note: Modified for dynamic num of classes
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, eos_coef, losses, num_keypoints=17):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_keypoints: number of keypoints for RLE loss sigma parameters.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        # Learnable per-keypoint sigma for RLE loss (one sigma per keypoint, shared across x and y)
        # Initialized to 1.0 (log(1) = 0), so initial RLE loss ≈ L1 loss
        self.log_sigma = nn.Parameter(torch.zeros(num_keypoints))

    def loss_labels(self, outputs, targets, indices, num_boxes, num_classes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        #empty_weight = torch.ones(num_classes + 1).to(outputs['pred_logits'].device)
        #empty_weight[-1] = self.eos_coef # sigmoid focal loss does not use weights
        
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        
        # one-hot encoding here
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        '''
        target_classes = torch.full(src_logits.shape[:2], num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes = F.one_hot(target_classes)
        target_classes[idx] = target_classes_o
        '''
        target_classes = target_classes_o
        src_logits = src_logits[idx]

        loss_ce = F.cross_entropy(src_logits, target_classes.float())
        #loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes.transpose(1, 2).float(), empty_weight)
        #loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes.float(), empty_weight) # use one-hot encoded probs instead of int classes
        #loss_ce = sigmoid_focal_loss(src_logits, target_classes.float(), alpha=0.25, gamma=2.0, reduction='mean') # use one-hot encoded probs instead of int classes
        losses = {'loss_ce': loss_ce}
        return losses
        
    def loss_human(self, outputs, targets, indices, num_boxes, num_classes, log=True): #HUMAN
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        empty_weight = torch.ones(1 + 1).to(outputs['human_logits'].device)
        empty_weight[-1] = self.eos_coef
        
        num_classes = 1
        
        assert 'human_logits' in outputs
        src_logits = outputs['human_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([torch.zeros(len(t['labels'])).long() for t, (_, J) in zip(targets, indices)]).to(src_logits.device)

        target_classes = torch.full(src_logits.shape[:2], num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_human = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        losses = {'loss_human': loss_human}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, num_classes): #quick fix for num_classes for loss_label
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, num_classes): #quick fix for num_classes for loss_label
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_keypoints(self, outputs, targets, indices, num_boxes, num_classes): #KEYPOINT
        """Compute the losses related to keypoints: RLE loss + visibility loss

           RLE (Residual Log-likelihood Estimation) models keypoint regression as a
           probabilistic problem with learnable per-keypoint uncertainty (sigma).

           For a Laplacian distribution, the negative log-likelihood is:
               NLL = |residual| / sigma + log(sigma)

           This allows the model to learn which keypoints are harder to predict
           (higher sigma) vs easier (lower sigma), providing adaptive weighting.

           targets dicts must contain the key "keypoints" containing a tensor of dim [nb_target_boxes, num_keypoints, 3]
           The keypoints are expected in format (x, y, visibility), normalized by the image size.
        """
        if 'pred_keypoints' not in outputs:
            return {}

        # Check if targets have keypoints
        if len(targets) == 0 or 'keypoints' not in targets[0]:
            return {}

        idx = self._get_src_permutation_idx(indices)

        # src_keypoints: [num_matched, num_keypoints, 3]
        src_keypoints = outputs['pred_keypoints'][idx]

        # target_keypoints: [num_matched, num_keypoints, 3]
        target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Visibility mask: only compute loss for visible keypoints
        # visibility > 0 means the keypoint is labeled (0=not labeled, 1=occluded, 2=visible)
        vis_mask = target_keypoints[..., 2] > 0  # [num_matched, num_keypoints]

        # === RLE Loss ===
        # Compute residuals (errors) for x and y coordinates
        residuals = torch.abs(src_keypoints[..., :2] - target_keypoints[..., :2])  # [num_matched, num_keypoints, 2]

        # Get sigma from learnable log_sigma (ensures sigma > 0)
        # sigma shape: [num_keypoints] -> broadcast to [1, num_keypoints, 1]
        sigma = torch.exp(self.log_sigma).view(1, -1, 1)  # [1, num_keypoints, 1]

        # RLE loss: |residual| / sigma + log(sigma)
        # The log(sigma) term prevents sigma from going to infinity
        rle_loss = residuals / sigma + self.log_sigma.view(1, -1, 1)  # [num_matched, num_keypoints, 2]

        # Apply visibility mask: only count loss for visible keypoints
        rle_loss = rle_loss * vis_mask.unsqueeze(-1)

        # Visibility prediction loss (binary cross-entropy)
        # Target: visibility > 0 means visible/labeled
        tgt_vis = (target_keypoints[..., 2] > 0).float()  # [num_matched, num_keypoints]
        src_vis = src_keypoints[..., 2]  # [num_matched, num_keypoints]
        loss_vis = F.binary_cross_entropy(src_vis, tgt_vis, reduction='none')

        # Compute losses
        num_visible = vis_mask.sum().clamp(min=1)

        losses = {}
        losses['loss_keypoints'] = rle_loss.sum() / num_visible
        losses['loss_keypoint_vis'] = loss_vis.mean()

        return losses

    def loss_keypoints_heatmap(self, outputs, targets, indices, num_boxes, num_classes):
        """Compute the heatmap-based keypoint loss (ViTPose-style MSE).

        Generates GT Gaussian heatmaps on-the-fly from normalized keypoint coordinates
        and computes MSE against predicted heatmaps. All persons in the image contribute
        to the same global heatmap channel (max-pooled).

        outputs must contain:
            'pred_heatmaps': [B, K, H_hm, W_hm] (softmax probability maps)
        targets must contain:
            'keypoints': list of [N_persons, K, 3] normalized (x, y, vis)
        """
        if 'pred_heatmaps' not in outputs:
            return {}
        if len(targets) == 0 or 'keypoints' not in targets[0]:
            return {}

        pred_heatmaps = outputs['pred_heatmaps']  # [B, K, H_hm, W_hm]
        B, K, H_hm, W_hm = pred_heatmaps.shape
        device = pred_heatmaps.device
        sigma = 2.0  # Gaussian sigma in heatmap pixels

        # Build GT heatmaps
        gt_heatmaps = torch.zeros_like(pred_heatmaps)

        # Coordinate grids in pixel space [H_hm, W_hm]
        ys = torch.arange(H_hm, device=device).float()
        xs = torch.arange(W_hm, device=device).float()
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H_hm, W_hm]

        for b, t in enumerate(targets):
            kps = t['keypoints']  # [N_persons, K, 3]
            for k in range(K):
                vis_mask = kps[:, k, 2] > 0  # [N_persons]
                if not vis_mask.any():
                    continue
                # GT keypoint pixel coords in heatmap space
                cx = kps[vis_mask, k, 0] * W_hm  # [N_vis]
                cy = kps[vis_mask, k, 1] * H_hm   # [N_vis]
                # Vectorized Gaussian: [N_vis, H_hm, W_hm]
                gaussians = torch.exp(
                    -((grid_x.unsqueeze(0) - cx.view(-1, 1, 1)) ** 2
                      + (grid_y.unsqueeze(0) - cy.view(-1, 1, 1)) ** 2)
                    / (2 * sigma ** 2)
                )
                # Max-pool over persons, clamp to [0, 1]
                gt_heatmaps[b, k] = gaussians.max(dim=0).values.clamp(0, 1)

        # Normalize GT heatmaps to sum to 1 (like softmax pred)
        gt_sum = gt_heatmaps.view(B, K, -1).sum(dim=-1, keepdim=True).view(B, K, 1, 1)
        gt_heatmaps = gt_heatmaps / (gt_sum + 1e-8)

        # MSE loss
        mse = F.mse_loss(pred_heatmaps, gt_heatmaps)

        return {'loss_keypoints': mse}

    def loss_masks(self, outputs, targets, indices, num_boxes, num_classes): #quick fix for num_classes for loss_label
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, num_classes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'human': self.loss_human, #HUMAN
            'keypoints': self.loss_keypoints, #KEYPOINT
            'keypoints_heatmap': self.loss_keypoints_heatmap, #HEATMAP
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        #return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
        return loss_map[loss](outputs, targets, indices, num_boxes, num_classes, **kwargs)

    def forward(self, outputs, targets, num_classes):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, num_classes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
        
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, imgsize, human_conf=0.7, Aaug=None, thresh=0.25):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_human, out_logits, out_bbox = outputs['human_logits'], outputs['pred_logits'], outputs['pred_boxes']

        if Aaug is not None:
            out_logits = out_logits @ Aaug

        human_prob = F.softmax(out_human, -1)
        human_scores, human_labels = human_prob[...,].max(-1)
        
        prob = out_logits
        
        boxes = box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([imgsize[1], imgsize[0], imgsize[1], imgsize[0]]).to(boxes.device)
        boxes = boxes * scale_fct

        results = []
        bs = out_human.shape[0]
        for i in range(bs):
            human_idx = torch.where(human_labels[i] == 0) # obtain boxes where human is detected
            human_scores_kept = human_scores[i][human_idx] # filter boxes where human is detected
            prob_kept = prob[i][human_idx]
            boxes_kept = boxes[i][human_idx]
            
            human_idx = torch.where(human_scores_kept >= human_conf) # obtain boxes where human conf > thresh (default=0.7)
            human_scores_kept = human_scores_kept[human_idx] # filter boxes where human is detected
            prob_kept = prob_kept[human_idx]
            boxes_kept = boxes_kept[human_idx]
            
            human_idx = batched_nms(boxes_kept, human_scores_kept, torch.zeros(len(human_scores_kept)).to(human_scores_kept.device), 0.5) # extra NMS
            human_scores_kept = human_scores_kept[human_idx] # filter boxes where human is detected
            prob_kept = prob_kept[human_idx]
            boxes_kept = boxes_kept[human_idx].int()

            final_scores = []
            final_labels = []
            finalboxes = []
            for i in range(len(human_idx)):
                box = boxes_kept[i]
                gt = torch.where(prob_kept[i] >= thresh)[0]
                gt_conf = (prob_kept[i][gt] + 1) / 2
                final_scores.extend(gt_conf)
                final_labels.extend(gt)
                for _ in range(len(gt)):
                    finalboxes.append(box)
            final_scores = torch.stack(final_scores) if len(final_scores) != 0 else torch.empty(0)
            final_labels = torch.stack(final_labels) if len(final_labels) != 0 else torch.empty(0)
            finalboxes = torch.stack(finalboxes) if len(finalboxes) != 0 else torch.empty(0)
            
            results.append({'scores': final_scores,
                            'labels': final_labels,
                            'boxes': finalboxes})

        return results

class PostProcessViz(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, imgsize, human_conf=0.7, Aaug=None, thresh=0.25):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_human, out_logits, out_bbox = outputs['human_logits'], outputs['pred_logits'], outputs['pred_boxes']

        if Aaug is not None:
            out_logits = out_logits @ Aaug

        human_prob = F.softmax(out_human, -1)
        human_scores, human_labels = human_prob[...,].max(-1)
        
        #prob = F.sigmoid(out_logits)
        prob = out_logits
        #scores, labels = prob.max(-1)
        
        boxes = box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([imgsize[1], imgsize[0], imgsize[1], imgsize[0]]).to(boxes.device)
        boxes = boxes * scale_fct

        results = []
        bs = out_human.shape[0]
        for i in range(bs):
            human_idx = torch.where(human_labels[i] == 0) # obtain boxes where human is detected
            human_scores_kept = human_scores[i][human_idx] # filter boxes where human is detected
            #scores_kept = scores[i][human_idx] # obtain boxes where human is detected
            #labels_kept = labels[i][human_idx]
            prob_kept = prob[i][human_idx]
            boxes_kept = boxes[i][human_idx]
            
            human_idx = torch.where(human_scores_kept >= human_conf) # obtain boxes where human conf > thresh (default=0.7)
            human_scores_kept = human_scores_kept[human_idx] # filter boxes where human is detected
            #scores_kept = scores_kept[human_idx] # obtain boxes where human is detected
            #labels_kept = labels_kept[human_idx]
            prob_kept = prob_kept[human_idx]
            boxes_kept = boxes_kept[human_idx]
            
            human_idx = batched_nms(boxes_kept, human_scores_kept, torch.zeros(len(human_scores_kept)).to(human_scores_kept.device), 0.5) # extra NMS
            human_scores_kept = human_scores_kept[human_idx] # filter boxes where human is detected
            #scores_kept = scores_kept[human_idx] # obtain boxes where human is detected
            #labels_kept = labels_kept[human_idx]
            prob_kept = prob_kept[human_idx]
            boxes_kept = boxes_kept[human_idx].int()

            final_scores = []
            final_labels = []
            finalboxes = []
            for i in range(len(human_idx)):
                box = boxes_kept[i]
                gt = torch.where(prob_kept[i] >= thresh)[0]
                gt_conf = (prob_kept[i][gt] + 1) / 2
                final_scores.append(gt_conf)
                final_labels.append(gt)
                finalboxes.append(box)
            #final_scores = torch.stack(final_scores)
            #final_labels = torch.stack(final_labels)
            #finalboxes = torch.stack(finalboxes)

            results.append({'scores': final_scores,
                            'labels': final_labels,
                            'boxes': finalboxes})

        return results


class PostProcessPose(nn.Module):
    """This module converts the model's pose output into a usable format.

    Only processes human detection and keypoints - no action class dependencies.
    """
    @torch.no_grad()
    def forward(self, outputs, imgsize, human_conf=0.7, keypoint_conf=0.5):
        """Perform pose post-processing.

        Parameters:
            outputs: raw outputs of the model containing:
                - human_logits: [B, num_queries, 2] human detection scores
                - pred_boxes: [B, num_queries, 4] bounding boxes (cxcywh normalized)
                - pred_keypoints: [B, num_queries, num_keypoints, 3] keypoints (x, y, vis)
            imgsize: (height, width) of the original image
            human_conf: confidence threshold for human detection
            keypoint_conf: confidence threshold for keypoint visibility

        Returns:
            List of dicts per batch, each containing:
                - scores: human detection scores
                - boxes: bounding boxes in xyxy format
                - keypoints: keypoints scaled to image size
        """
        out_human = outputs['human_logits']
        out_bbox = outputs['pred_boxes']
        out_keypoints = outputs.get('pred_keypoints', None)
        out_heatmaps = outputs.get('pred_heatmaps', None)  # [B, K, H_hm, W_hm] for heatmap model

        # Human detection scores
        human_prob = F.softmax(out_human, -1)
        human_scores, human_labels = human_prob.max(-1)

        # Scale boxes to image size
        boxes = box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([imgsize[1], imgsize[0], imgsize[1], imgsize[0]]).to(boxes.device)
        boxes = boxes * scale_fct

        # Scale keypoints to image size if available (regression model)
        if out_keypoints is not None and out_heatmaps is None:
            keypoints_scaled = out_keypoints.clone()
            keypoints_scaled[..., 0] = out_keypoints[..., 0] * imgsize[1]  # x * width
            keypoints_scaled[..., 1] = out_keypoints[..., 1] * imgsize[0]  # y * height
        else:
            keypoints_scaled = None

        results = []
        bs = out_human.shape[0]

        for i in range(bs):
            # Filter: human detected (label == 0)
            human_idx = torch.where(human_labels[i] == 0)[0]
            human_scores_kept = human_scores[i][human_idx]
            boxes_kept = boxes[i][human_idx]
            keypoints_kept = keypoints_scaled[i][human_idx] if keypoints_scaled is not None else None

            # Filter: human confidence threshold
            conf_idx = torch.where(human_scores_kept >= human_conf)[0]
            human_scores_kept = human_scores_kept[conf_idx]
            boxes_kept = boxes_kept[conf_idx]
            if keypoints_kept is not None:
                keypoints_kept = keypoints_kept[conf_idx]

            # NMS
            nms_idx = batched_nms(
                boxes_kept, human_scores_kept,
                torch.zeros(len(human_scores_kept)).to(human_scores_kept.device), 0.5
            )
            human_scores_kept = human_scores_kept[nms_idx]
            boxes_kept = boxes_kept[nms_idx].int()
            if keypoints_kept is not None:
                keypoints_kept = keypoints_kept[nms_idx]
                # Zero out low-confidence keypoint visibility
                keypoints_kept = keypoints_kept.clone()
                low_conf = keypoints_kept[..., 2] < keypoint_conf
                keypoints_kept[..., 2][low_conf] = 0

            # Heatmap model: extract per-person keypoints from global heatmaps
            if out_heatmaps is not None:
                heatmaps_i = out_heatmaps[i]  # [K, H_hm, W_hm]
                K, H_hm, W_hm = heatmaps_i.shape
                img_h, img_w = imgsize

                # Scale factors from image to heatmap coordinates
                scale_x = W_hm / img_w
                scale_y = H_hm / img_h

                person_kps = []
                for box in boxes_kept:
                    # Clamp bbox to image bounds and convert to heatmap coords
                    x1 = max(int(box[0].item() * scale_x), 0)
                    y1 = max(int(box[1].item() * scale_y), 0)
                    x2 = min(int(box[2].item() * scale_x) + 1, W_hm)
                    y2 = min(int(box[3].item() * scale_y) + 1, H_hm)

                    # Clamp to valid heatmap region
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, W_hm), min(y2, H_hm)

                    kps = torch.zeros(K, 3, device=heatmaps_i.device)
                    if x2 > x1 and y2 > y1:
                        roi = heatmaps_i[:, y1:y2, x1:x2]  # [K, roi_h, roi_w]
                        roi_h, roi_w = roi.shape[1], roi.shape[2]

                        # Argmax within ROI
                        roi_flat = roi.reshape(K, -1)
                        max_vals, flat_idx = roi_flat.max(dim=-1)  # [K]

                        row_idx = flat_idx // roi_w + y1
                        col_idx = flat_idx % roi_w + x1

                        # Convert back to image pixel coordinates
                        kps[:, 0] = col_idx.float() / scale_x  # x in image space
                        kps[:, 1] = row_idx.float() / scale_y  # y in image space
                        kps[:, 2] = max_vals  # confidence from heatmap peak

                    person_kps.append(kps)

                if len(person_kps) > 0:
                    keypoints_kept = torch.stack(person_kps, dim=0)  # [N_persons, K, 3]
                    # Zero out low-confidence keypoints
                    low_conf = keypoints_kept[..., 2] < keypoint_conf
                    keypoints_kept = keypoints_kept.clone()
                    keypoints_kept[..., 2][low_conf] = 0
                else:
                    # No detections: return empty tensor
                    keypoints_kept = torch.zeros((0, K, 3), device=heatmaps_i.device)

            result = {
                'scores': human_scores_kept,
                'boxes': boxes_kept,
            }
            if keypoints_kept is not None:
                result['keypoints'] = keypoints_kept

            results.append(result)

        return results


# COCO keypoint names for reference
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO skeleton connections for visualization (1-indexed)
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # legs
    [6, 12], [7, 13],  # torso to hips
    [6, 7],  # shoulders
    [6, 8], [7, 9], [8, 10], [9, 11],  # arms
    [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]  # face to shoulders
]
