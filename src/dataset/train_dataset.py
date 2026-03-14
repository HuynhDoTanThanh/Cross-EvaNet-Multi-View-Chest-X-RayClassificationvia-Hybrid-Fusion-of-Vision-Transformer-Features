"""Training dataset for multi-view chest X-ray studies."""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.constants import LABEL_COLUMNS


class MultiViewXRayDataset(Dataset):
    """Dataset that groups images by (Patient_ID, Study) and yields num_views per sample."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform,
        num_views: int = 3,
        label_columns: Optional[List[str]] = None,
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform
        self.num_views = num_views
        self.label_columns = label_columns or LABEL_COLUMNS

        self.study_map = (
            df.groupby(["Patient_ID", "Study"])["Image_name"].apply(list).to_dict()
        )
        self.study_keys = list(self.study_map.keys())
        self.df_labels = df.groupby(["Patient_ID", "Study"])[self.label_columns].first()

    def __len__(self) -> int:
        return len(self.study_keys)

    def __getitem__(self, idx: int):
        key = self.study_keys[idx]
        image_names = self.study_map[key]
        labels = torch.from_numpy(
            self.df_labels.loc[key].to_numpy(dtype=np.float32).copy()
        )

        start_index = np.random.randint(0, len(image_names))
        selected_names = []
        i = start_index
        while len(selected_names) < self.num_views:
            img_name = image_names[i % len(image_names)]
            selected_names.append(img_name)
            i += 1

        tensor_stack = []
        for name in selected_names:
            p = os.path.join(self.image_dir, str(name))
            img = Image.open(p).convert("RGB")
            img = self.transform(img)
            tensor_stack.append(img)

        images = torch.stack(tensor_stack, dim=0)
        return images, labels
