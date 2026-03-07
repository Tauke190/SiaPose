"""SIA pose models - multiple variants with different architectures."""
from .sia_pose_simple import SIA_POSE_SIMPLE
from .sia_pose_coco_decoder import SIA_POSE_SIMPLE_DEC
from .sia_pose_coco_roi import SIA_POSE_SIMPLE_DEC_ROI, extract_roi_features_aligned
from .sia_pose_coco_roi_best_so_far import SIA_POSE_SIMPLE_DEC_ROI_BEST
from .sia_pos_eomt import SIA_POSE_EOMT, get_sia_pose_eomt

__all__ = [
    "SIA_POSE_SIMPLE",
    "SIA_POSE_SIMPLE_DEC",
    "SIA_POSE_SIMPLE_DEC_ROI",
    "SIA_POSE_SIMPLE_DEC_ROI_BEST",
    "SIA_POSE_EOMT",
    "extract_roi_features_aligned",
    "get_sia_pose_eomt",
]
