"""Inference and submission building."""

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch

from src.constants import LABEL_COLUMNS, NUM_LABELS


def run_inference(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    use_xla: bool = False,
) -> List[dict]:
    """Run model on test loader; returns list of {study_key, probs}."""
    model.eval()
    results = []
    dtype = torch.bfloat16 if use_xla else torch.float16
    device_type = "xla" if use_xla else "cuda"

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["images"].to(device)
            study_keys = batch["study_key"]
            with torch.autocast(device_type=device_type, dtype=dtype):
                output = model(imgs)
                logits = output if isinstance(output, torch.Tensor) else output.get("logits", output)
                probs = torch.sigmoid(logits)
            probs_np = probs.float().cpu().numpy()
            for key, prob in zip(study_keys, probs_np):
                results.append({"study_key": key, "probs": prob})
    return results


def aggregate_study_preds(
    results: List[dict],
    num_labels: int = NUM_LABELS,
) -> pd.DataFrame:
    """Average probabilities per study (sliding-window merge)."""
    df_results = pd.DataFrame(results)
    prob_cols = [f"p_{i}" for i in range(num_labels)]
    df_probs = pd.DataFrame(df_results["probs"].tolist(), columns=prob_cols)
    df_results = pd.concat([df_results[["study_key"]], df_probs], axis=1)
    final = df_results.groupby("study_key")[prob_cols].mean().reset_index()
    return final


def build_submission(
    test_dir: str,
    final_study_preds: pd.DataFrame,
    output_path: str = "submission.csv",
    label_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build per-image submission CSV from study-level predictions."""
    label_columns = label_columns or LABEL_COLUMNS
    prob_cols = [f"p_{i}" for i in range(len(label_columns))]
    all_files = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )
    pred_lookup = final_study_preds.set_index("study_key")[prob_cols].to_dict("index")
    submission_rows = []
    for img_name in all_files:
        parts = img_name.split("_")
        study_key = f"{parts[0]}_{parts[1]}"
        probs = list(pred_lookup[study_key].values())
        submission_rows.append([img_name] + probs)
    submission_df = pd.DataFrame(
        submission_rows, columns=["Image_name"] + label_columns
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return submission_df
