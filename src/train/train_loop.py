"""Training loop, evaluation, and scheduler."""

import math
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.constants import NUM_LABELS


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    cfg,
    use_xla: bool = False,
) -> float:
    """Run one training epoch. Supports TPU (XLA) or GPU."""
    model.train()
    total_loss = 0.0
    step = 0
    xm = None
    if use_xla:
        try:
            import torch_xla.core.xla_model as xm
        except ImportError:
            use_xla = False

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        dtype = torch.bfloat16 if use_xla else torch.float16
        device_type = "xla" if use_xla else "cuda"
        with torch.autocast(device_type=device_type, dtype=dtype):
            logits = model(imgs)
            loss1 = criterion(logits, labels)
            loss2 = criterion(logits, labels)
            loss = (loss1 + loss2) / cfg.grad_accum_steps

        loss.backward()
        step += 1

        if step % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            if use_xla and xm is not None:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        total_loss += loss.item() * cfg.grad_accum_steps

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_labels: int = NUM_LABELS,
    use_xla: bool = False,
) -> Tuple[float, list]:
    """Compute macro ROC-AUC and per-class AUC."""
    model.eval()
    all_logits = []
    all_labels = []
    dtype = torch.bfloat16 if use_xla else torch.float16
    device_type = "xla" if use_xla else "cuda"

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device_type, dtype=dtype):
            outputs = model(imgs)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.get("logits", outputs)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.float().cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-all_logits))

    per_class_auc = []
    for c in range(num_labels):
        y_true = all_labels[:, c]
        y_pred = probs[:, c]
        if np.unique(y_true).size < 2:
            per_class_auc.append(np.nan)
        else:
            per_class_auc.append(roc_auc_score(y_true, y_pred))
    macro_auc = float(np.nanmean(per_class_auc))
    return macro_auc, per_class_auc


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """Cosine schedule with linear warmup."""
    total_steps = cfg.epochs * math.ceil(steps_per_epoch / cfg.grad_accum_steps)
    warmup_steps = int(cfg.warmup_pct * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * (1.0 + math.cos(math.pi * progress)) / 2.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
