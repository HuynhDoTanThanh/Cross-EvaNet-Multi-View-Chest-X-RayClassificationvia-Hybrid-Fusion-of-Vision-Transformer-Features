# Cross-EvaNet: Multi-View Chest X-RayClassificationvia Hybrid Fusion of Vision Transformer Features

Multi-view chest X-ray multi-label classification using a **Triple-Branch EVA** architecture: frozen single-view backbone, multi-view fusion backbone, and a fusion head. Designed for TPU (Kaggle/Colab) and GPU.

## Competition

- **[Grand X-Ray Slam: Division A](https://www.kaggle.com/competitions/grand-xray-slam-division-a)**  
- **[Grand X-Ray Slam: Division B](https://www.kaggle.com/competitions/grand-xray-slam-division-b)**

## Project structure

```
chexray-multiview/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                 # Optional YAML configs (future)
├── src/
│   ├── __init__.py
│   ├── constants.py         # LABEL_COLUMNS, NUM_LABELS
│   ├── config.py           # TrainConfig dataclass
│   ├── dataset/
│   │   ├── train_dataset.py # MultiViewXRayDataset
│   │   ├── test_dataset.py  # TestMultiViewDataset (sliding window)
│   │   ├── transforms.py   # Augmentations, build_transforms
│   │   └── loaders.py      # make_loaders
│   ├── models/
│   │   ├── eva_backbone.py  # EVA-X, checkpoint loading
│   │   ├── multiview.py    # MultiImageHybridEVA
│   │   └── triple_branch.py # TripleBranchEVA, build_*
│   ├── losses.py           # FocalLoss, ASL, APLLoss
│   ├── train/
│   │   └── train_loop.py    # train_one_epoch, evaluate, build_scheduler
│   └── inference.py        # run_inference, aggregate_study_preds, build_submission
├── scripts/
│   ├── train.py            # Training entrypoint
│   └── inference.py        # Inference + submission CSV
└── notebooks/              # Optional: original notebook
```

## Setup

```bash
cd chexray-multiview
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

For **TPU** (Kaggle/Colab), install `torch_xla` in that environment.

## Data

- **Train CSV**: `Patient_ID`, `Study`, `Image_name`, and 14 label columns (e.g. Atelectasis, Cardiomegaly, …).
- **Train dir**: Directory containing training images (filenames as in `Image_name`).
- **Test dir**: Directory of test images; filenames assumed `PatientID_StudyID_*.jpg` for grouping.

## Training

**GPU:**

```bash
python scripts/train.py \
  --train-csv path/to/train_mv.csv \
  --train-dir path/to/train_images \
  [--pretrained path/to/single_view.pth] \
  [--pretrained-2 path/to/multiview.pth] \
  --save-path outputs/vit_l16_multiview.pth \
  --epochs 8 --batch-size 10 --num-views 2
```

**TPU (e.g. Kaggle):**

```bash
python scripts/train.py \
  --train-csv /kaggle/input/.../train_mv.csv \
  --train-dir /kaggle/input/.../train \
  --pretrained /kaggle/input/.../vit_l16_singleview.pth \
  --pretrained-2 /kaggle/input/.../vit_l16_multiview.pth \
  --save-path /kaggle/working/vit_l16_multiview.pth \
  --use-tpu
```

## Inference

```bash
python scripts/inference.py \
  --checkpoint outputs/vit_l16_multiview.pth \
  --test-dir path/to/test_images \
  --output submission.csv
```

Use `--use-tpu` when running on TPU.

## Labels (14 classes)

Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices.

## License

Use according to your data and competition rules.
