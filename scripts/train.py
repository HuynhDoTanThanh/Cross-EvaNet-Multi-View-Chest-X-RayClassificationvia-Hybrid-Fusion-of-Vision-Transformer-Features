#!/usr/bin/env python3
"""
Training script for multi-view CheXray (Triple-Branch EVA).
Supports TPU (Kaggle/Colab) via torch_xla and GPU.
Usage:
  # TPU (set TPU env or run on Kaggle)
  python scripts/train.py --train-csv data/train_mv.csv --train-dir data/train --use-tpu

  # GPU
  python scripts/train.py --train-csv data/train_mv.csv --train-dir data/train
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Optional: set before other imports for TF/XLA
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_USE_BF16", "1")

from src.config import TrainConfig
from src.constants import LABEL_COLUMNS, NUM_LABELS, seed_everything
from src.dataset import build_transforms, make_loaders
from src.losses import APLLoss
from src.models import build_triple_branch_model
from src.train import build_scheduler, evaluate, train_one_epoch


def parse_args():
    p = argparse.ArgumentParser(description="Train multi-view CheXray model")
    p.add_argument("--train-csv", required=True, help="Path to train CSV (Patient_ID, Study, Image_name, ...)")
    p.add_argument("--train-dir", required=True, help="Directory containing training images")
    p.add_argument("--pretrained", default=None, help="Single-view EVA checkpoint path")
    p.add_argument("--pretrained-2", default=None, help="Multi-view EVA checkpoint path")
    p.add_argument("--save-path", default="outputs/vit_l16_multiview.pth", help="Where to save best checkpoint")
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--num-views", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--use-tpu", action="store_true", help="Use TPU (torch_xla)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        pretrained=args.pretrained,
        pretrained_2=args.pretrained_2,
        save_path=args.save_path,
        img_size=args.img_size,
        num_views=args.num_views,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
    )

    use_xla = args.use_tpu
    if use_xla:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        device = xm.xla_device()
        xm.master_print("Using device: {}".format(device))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

    # Seed
    seed_everything(cfg.seed)

    # Data
    label_dtypes = {c: "float32" for c in LABEL_COLUMNS}
    df = pd.read_csv(cfg.train_csv, dtype=label_dtypes)
    df[LABEL_COLUMNS] = df[LABEL_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    patients = df["Patient_ID"].unique()
    np.random.shuffle(patients)
    split_idx = int(len(patients) * (1 - cfg.val_split))
    train_patients = patients[:split_idx]
    val_patients = patients[split_idx:]
    train_df = df[df["Patient_ID"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["Patient_ID"].isin(val_patients)].reset_index(drop=True)
    if use_xla:
        xm.master_print(f"Train Rows: {len(train_df)}, Val Rows: {len(val_df)}")
    else:
        print(f"Train Rows: {len(train_df)}, Val Rows: {len(val_df)}")

    model = build_triple_branch_model(NUM_LABELS, cfg)
    model = model.to(device)

    train_tf, val_tf = build_transforms(cfg.img_size)
    train_loader, val_loader = make_loaders(train_df, val_df, cfg, train_tf, val_tf)
    if use_xla:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = APLLoss()
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    best_auc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        msg = f"\n===== Epoch {epoch}/{cfg.epochs} ====="
        if use_xla:
            xm.master_print(msg)
        else:
            print(msg)
        total_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, cfg, use_xla=use_xla
        )
        current_lr = optimizer.param_groups[0]["lr"]
        if use_xla:
            xm.master_print(f"total_loss: {total_loss:.4f} | LR: {current_lr:.6f}")
        else:
            print(f"total_loss: {total_loss:.4f} | LR: {current_lr:.6f}")

        val_auc, per_class_auc = evaluate(model, val_loader, device, use_xla=use_xla)
        if use_xla:
            xm.master_print(f"val macro ROC-AUC: {val_auc:.4f}")
            xm.master_print(f"per_class_auc: {per_class_auc}")
        else:
            print(f"val macro ROC-AUC: {val_auc:.4f}")
            print("per_class_auc:", per_class_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            if use_xla:
                xm.master_print(f"New best AUC {best_auc:.4f}. Saving...")
                xm.save(model.state_dict(), cfg.save_path)
            else:
                print(f"New best AUC {best_auc:.4f}. Saving...")
                torch.save(model.state_dict(), cfg.save_path)

    print("Training finished. Best AUC:", best_auc)


if __name__ == "__main__":
    main()
