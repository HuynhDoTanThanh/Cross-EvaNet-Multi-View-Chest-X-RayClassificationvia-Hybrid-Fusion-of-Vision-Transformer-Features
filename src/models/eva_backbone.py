"""EVA backbone and checkpoint loading utilities."""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import resample_abs_pos_embed, resample_patch_embed
from timm.models.eva import Eva


def checkpoint_filter_fn(
    state_dict,
    model,
    interpolation="bicubic",
    antialias=True,
):
    """Convert patch embedding / pos_embed for compatibility."""
    out_dict = {}
    state_dict = state_dict.get("model_ema", state_dict)
    state_dict = state_dict.get("model", state_dict)
    state_dict = state_dict.get("module", state_dict)
    state_dict = state_dict.get("state_dict", state_dict)
    if "visual.trunk.pos_embed" in state_dict:
        prefix = "visual.trunk."
    elif "visual.pos_embed" in state_dict:
        prefix = "visual."
    else:
        prefix = ""
    mim_weights = prefix + "mask_token" in state_dict
    no_qkv = prefix + "blocks.0.attn.q_proj.weight" in state_dict
    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue
        if "rope" in k:
            continue
        if "patch_embed.proj.weight" in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v, (H, W), interpolation=interpolation, antialias=antialias, verbose=True
                )
        elif k == "pos_embed" and v.shape[1] != model.pos_embed.shape[1]:
            num_prefix_tokens = (
                0
                if getattr(model, "no_embed_class", False)
                else getattr(model, "num_prefix_tokens", 1)
            )
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        k = k.replace("mlp.ffn_ln", "mlp.norm")
        k = k.replace("attn.inner_attn_ln", "attn.norm")
        k = k.replace("mlp.w12", "mlp.fc1")
        k = k.replace("mlp.w1", "mlp.fc1_g")
        k = k.replace("mlp.w2", "mlp.fc1_x")
        k = k.replace("mlp.w3", "mlp.fc2")
        if no_qkv:
            k = k.replace("q_bias", "q_proj.bias")
            k = k.replace("v_bias", "v_proj.bias")
        if mim_weights and k in (
            "mask_token",
            "lm_head.weight",
            "lm_head.bias",
            "norm.weight",
            "norm.bias",
        ):
            if k == "norm.weight" or k == "norm.bias":
                k = k.replace("norm", "fc_norm")
            else:
                continue
        out_dict[k] = v
    return out_dict


def load_with_pos_embed_interpolation(
    model: nn.Module, checkpoint_path: str, strict: bool = False
) -> nn.Module:
    """Load checkpoint and interpolate pos_embed if shape mismatch."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    key = "model.pos_embed"
    if key not in state_dict and "pos_embed" in state_dict:
        key = "pos_embed"
    if key in state_dict:
        pos_embed_checkpoint = state_dict[key]
        pos_embed_model = model.state_dict().get(key, None)
        if pos_embed_model is not None and pos_embed_checkpoint.shape != pos_embed_model.shape:
            print(f"Resizing {key}: {pos_embed_checkpoint.shape} -> {pos_embed_model.shape}")
            cls_token = pos_embed_checkpoint[:, 0:1, :]
            img_tokens = pos_embed_checkpoint[:, 1:, :]
            n_tokens_old = img_tokens.shape[1]
            grid_size_old = int(math.sqrt(n_tokens_old))
            n_tokens_new = pos_embed_model.shape[1] - 1
            grid_size_new = int(math.sqrt(n_tokens_new))
            img_tokens = img_tokens.reshape(
                1, grid_size_old, grid_size_old, -1
            ).permute(0, 3, 1, 2)
            img_tokens = F.interpolate(
                img_tokens,
                size=(grid_size_new, grid_size_new),
                mode="bicubic",
                align_corners=False,
            )
            img_tokens = img_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((cls_token, img_tokens), dim=1)
            state_dict[key] = new_pos_embed
    msg = model.load_state_dict(state_dict, strict=strict)
    print("Load status:", msg)
    return model


class EVA_X(Eva):
    """EVA with explicit forward_features / forward_head for integration."""

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = (
                x[:, self.num_prefix_tokens :].mean(dim=1)
                if self.global_pool == "avg"
                else x[:, 0]
            )
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def eva_x_base_patch16(
    pretrained: Union[bool, str] = False,
    drop_path_rate: float = 0.0,
    img_size: int = 448,
    num_classes: int = 14,
) -> EVA_X:
    """Build EVA-X base 448 patch16 with optional pretrained weights."""
    model = EVA_X(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(28, 28),
        drop_path_rate=drop_path_rate,
    )
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    if isinstance(pretrained, str):
        print(f"Loading pretrained weights from: {pretrained}")
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        msg = model.load_state_dict(ckpt, strict=False)
        print(msg)
    return model
