"""Training and inference configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    """Configuration for multi-view CheXray training."""

    # Data
    train_csv: str = "train_mv.csv"
    train_dir: str = ""
    val_split: float = 0.1

    # Checkpoints
    pretrained: Optional[str] = None  # Single-view EVA weights
    pretrained_2: Optional[str] = None  # Multi-view EVA weights
    save_path: str = "outputs/vit_l16_multiview.pth"

    # Model
    img_size: int = 448
    num_views: int = 2
    drop_path_rate: float = 0.2

    # Training
    batch_size: int = 10
    num_workers: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 8
    warmup_pct: float = 0.05
    grad_accum_steps: int = 4
    clip_grad: float = 1.0
    use_focal: bool = True
    seed: int = 1337

    def __post_init__(self) -> None:
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
