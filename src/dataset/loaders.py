"""DataLoader factory."""

from torch.utils.data import DataLoader

from .train_dataset import MultiViewXRayDataset
from .transforms import build_transforms


def make_loaders(train_df, val_df, cfg, train_tf, val_tf):
    """Build train and val DataLoaders for multi-view dataset."""
    train_ds = MultiViewXRayDataset(
        train_df, cfg.train_dir, transform=train_tf, num_views=cfg.num_views
    )
    val_ds = MultiViewXRayDataset(
        val_df, cfg.train_dir, transform=val_tf, num_views=cfg.num_views
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )
    return train_dl, val_dl
