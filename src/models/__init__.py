"""Model definitions: EVA backbone, multi-view fusion, triple-branch, VAAF."""

from .eva_backbone import EVA_X, eva_x_base_patch16, load_with_pos_embed_interpolation
from .vaaf import ViewAwareAttentionFusion
from .multiview import MultiImageHybridEVA
from .triple_branch import TripleBranchEVA, build_triple_branch_model

__all__ = [
    "EVA_X",
    "eva_x_base_patch16",
    "load_with_pos_embed_interpolation",
    "ViewAwareAttentionFusion",
    "MultiImageHybridEVA",
    "TripleBranchEVA",
    "build_triple_branch_model",
]
