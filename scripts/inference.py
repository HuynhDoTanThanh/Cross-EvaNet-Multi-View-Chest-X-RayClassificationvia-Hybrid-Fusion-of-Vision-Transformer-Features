#!/usr/bin/env python3
"""
Inference script: load checkpoint, run on test images, write submission CSV.
Usage:
  python scripts/inference.py \
    --checkpoint outputs/vit_l16_multiview.pth \
    --test-dir data/test \
    --output submission.csv \
    [--use-tpu]
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.constants import NUM_LABELS
from src.dataset import TestMultiViewDataset, build_transforms
from src.inference import aggregate_study_preds, build_submission, run_inference
from src.models import build_triple_branch_model


def parse_args():
    p = argparse.ArgumentParser(description="Run inference and build submission")
    p.add_argument("--checkpoint", required=True, help="Path to model state_dict (.pth)")
    p.add_argument("--test-dir", required=True, help="Directory of test images (*.jpg)")
    p.add_argument("--output", default="submission.csv", help="Output submission CSV path")
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--num-views", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--use-tpu", action="store_true", help="Use TPU (torch_xla)")
    return p.parse_args()


def main():
    args = parse_args()
    use_xla = args.use_tpu
    if use_xla:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        device = xm.xla_device()
        xm.master_print("Using device: {}".format(device))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

    cfg = TrainConfig(
        img_size=args.img_size,
        num_views=args.num_views,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    _, val_tf = build_transforms(cfg.img_size)
    test_ds = TestMultiViewDataset(
        args.test_dir, transform=val_tf, num_views=cfg.num_views, img_size=cfg.img_size
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    if use_xla:
        test_loader = pl.MpDeviceLoader(test_loader, device)

    model = build_triple_branch_model(NUM_LABELS, cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    if use_xla:
        xm.master_print("Starting inference...")
    else:
        print("Starting inference...")
    results = run_inference(model, test_loader, device, use_xla=use_xla)
    final_preds = aggregate_study_preds(results, num_labels=NUM_LABELS)
    build_submission(args.test_dir, final_preds, output_path=args.output)
    print("Submission saved to", args.output)


if __name__ == "__main__":
    main()
