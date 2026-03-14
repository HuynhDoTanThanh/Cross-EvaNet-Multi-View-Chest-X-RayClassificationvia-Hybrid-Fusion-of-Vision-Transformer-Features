"""Test/inference dataset for multi-view chest X-ray studies."""

import os
from typing import Any, Dict, List

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TestMultiViewDataset(Dataset):
    """Test dataset that discovers studies from a directory and builds sliding-window samples."""

    def __init__(
        self,
        root_dir: str,
        transform,
        num_views: int = 3,
        img_size: int = 448,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.num_views = num_views
        self.img_size = img_size

        all_files = sorted(
            [f for f in os.listdir(root_dir) if f.lower().endswith(".jpg")]
        )
        self.study_map: Dict[str, List[str]] = {}
        for f in all_files:
            parts = f.split("_")
            if len(parts) >= 2:
                study_key = f"{parts[0]}_{parts[1]}"
            else:
                study_key = f
            if study_key not in self.study_map:
                self.study_map[study_key] = []
            self.study_map[study_key].append(f)

        self.inference_samples: List[Dict[str, Any]] = []
        for study_key, images in self.study_map.items():
            images = sorted(images)
            n_imgs = len(images)

            if n_imgs < self.num_views:
                self.inference_samples.append(
                    {"study_key": study_key, "images": images, "needs_repeat": True}
                )
            else:
                for i in range(n_imgs - self.num_views + 1):
                    window = images[i : i + self.num_views]
                    self.inference_samples.append(
                        {
                            "study_key": study_key,
                            "images": window,
                            "needs_repeat": False,
                        }
                    )

    def __len__(self) -> int:
        return len(self.inference_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.inference_samples[idx]
        image_list = sample["images"]

        final_list: List[str] = []
        if sample["needs_repeat"]:
            i = 0
            while len(final_list) < self.num_views:
                final_list.append(image_list[i % len(image_list)])
                i += 1
        else:
            final_list = image_list

        tensor_stack = []
        for img_name in final_list:
            p = os.path.join(self.root_dir, img_name)
            try:
                img = Image.open(p).convert("RGB")
                img = self.transform(img)
                tensor_stack.append(img)
            except Exception:
                tensor_stack.append(torch.zeros((3, self.img_size, self.img_size)))

        images = torch.stack(tensor_stack, dim=0)
        return {"images": images, "study_key": sample["study_key"]}
