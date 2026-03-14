"""Image transforms and augmentation for chest X-ray."""

import random
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F_torch
from PIL import Image


class RandomGamma:
    """Randomly adjusts gamma (brightness/contrast curve)."""

    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.3), p: float = 0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            return F_torch.adjust_gamma(img, gamma)
        return img


class RandomCLAHE:
    """Applies CLAHE to PIL image."""

    def __init__(
        self,
        clip_limit: float = 4.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        p: float = 0.5,
    ):
        self.p = p
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img_np = np.array(img)
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
            )
            img_clahe = clahe.apply(gray)
            img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(img_clahe)
        return img


class RandomNoise:
    """Adds Gaussian noise to simulate scanner artifacts."""

    def __init__(
        self,
        mean: float = 0.0,
        std_range: Tuple[float, float] = (0.01, 0.05),
        p: float = 0.3,
    ):
        self.mean = mean
        self.std_range = std_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img_tensor = T.ToTensor()(img)
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(img_tensor) * std + self.mean
            img_tensor = torch.clamp(img_tensor + noise, 0, 1)
            return T.ToPILImage()(img_tensor)
        return img


def build_transforms(img_size: int):
    """Build train and validation transforms (dataset-specific mean/std)."""
    mean = (0.49185243, 0.49185243, 0.49185243)
    std = (0.28509309, 0.28509309, 0.28509309)

    train_tf = T.Compose([
        T.Resize(int(img_size * 1.15), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.85, 1.0),
            shear=5,
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomResizedCrop(
            size=img_size,
            ratio=(0.95, 1.05),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        RandomCLAHE(clip_limit=3.0, p=0.4),
        RandomGamma(gamma_range=(0.7, 1.3), p=0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        RandomNoise(p=0.3),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    val_tf = T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return train_tf, val_tf
