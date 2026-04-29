"""Triple-branch model: single-view + multi-view + fusion head."""

import torch
import torch.nn as nn

from .eva_backbone import eva_x_base_patch16
from .vaaf import ViewAwareAttentionFusion
from .multiview import MultiImageHybridEVA, build_multiview_model


def build_model_evax(num_classes: int, cfg):
    """Single-view EVA-X backbone (for triple-branch)."""
    pretrained = getattr(cfg, "pretrained", None)
    model = eva_x_base_patch16(
        pretrained=pretrained or False,
        drop_path_rate=cfg.drop_path_rate,
        img_size=cfg.img_size,
        num_classes=num_classes,
    )
    return model


class TripleBranchEVA(nn.Module):
    """Single-view (frozen) + multi-view + fusion head."""

    def __init__(
        self,
        single_view_model: nn.Module,
        multi_view_model: MultiImageHybridEVA,
        num_classes: int,
        embed_dim: int,
        freeze_single: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.single_model = single_view_model
        if freeze_single:
            print("🔒 Freezing Single View Branch (Weights & BN stats)...")
            for param in self.single_model.parameters():
                param.requires_grad = False
            self.single_model.eval()

        self.multi_model = multi_view_model
        self.fusion_head = ViewAwareAttentionFusion(
            embed_dim=embed_dim,
            num_labels=num_classes,
            num_heads=4,
            dropout=0.2,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "single_model"):
            is_frozen = not next(self.single_model.parameters()).requires_grad
            if is_frozen:
                self.single_model.eval()

    def forward_single_branch(self, x):
        features = self.single_model.forward_features(x)
        vec = self.single_model.forward_head(features, pre_logits=True)
        logits = self.single_model.head(vec)
        return vec, logits

    def forward_multi_branch(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size * 2, -1, x.shape[-2], x.shape[-1])
        backbone = getattr(
            self.multi_model, "backbone_fused", getattr(self.multi_model, "model", None)
        )
        feat = backbone.patch_embed(x_flat)
        feat, rot_pos_embed = backbone._pos_embed(feat)
        feat_fused = self.multi_model.format_multi_image_tokens(feat, batch_size)
        rope_fused = self.multi_model.adapt_rope_for_multiview(rot_pos_embed)
        for blk in backbone.blocks:
            feat_fused = blk(feat_fused, rope=rope_fused)
        feat_fused = backbone.norm(feat_fused)
        vec_fused = backbone.forward_head(feat_fused, pre_logits=True)
        return vec_fused

    def forward(self, x):
        img1 = x[:, 0]
        img2 = x[:, 1]
        if not next(self.single_model.parameters()).requires_grad:
            with torch.no_grad():
                vec_1, logits_1 = self.forward_single_branch(img1)
                vec_2, logits_2 = self.forward_single_branch(img2)
        else:
            vec_1, logits_1 = self.forward_single_branch(img1)
            vec_2, logits_2 = self.forward_single_branch(img2)
        avg_single_logits = (logits_1 + logits_2) / 2
        vec_fused = self.forward_multi_branch(x)
        logits_fusion = self.fusion_head(vec_1, vec_fused, vec_2)
        final_logits = avg_single_logits + logits_fusion
        return final_logits


def build_triple_branch_model(num_classes: int, cfg, embed_dim: int = 768):
    """Build full triple-branch model."""
    single_backbone = build_model_evax(num_classes, cfg)
    multi_view_model = build_multiview_model(num_classes, cfg)
    model = TripleBranchEVA(
        single_view_model=single_backbone,
        multi_view_model=multi_view_model,
        num_classes=num_classes,
        embed_dim=embed_dim,
        freeze_single=True,
    )
    return model
