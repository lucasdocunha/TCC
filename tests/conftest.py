from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest


@pytest.fixture
def tiny_phase1_dataset(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "phase1"
    split_dirs = {
        "train": "trainset",
        "val": "valset",
        "test": "testset",
    }

    for split, subdir in split_dirs.items():
        out_dir = dataset_root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(repo_root / "data" / "raw_min" / f"{split}.csv").head(16)
        for idx, row in df.reset_index(drop=True).iterrows():
            base = 48 + (idx * 11) % 160
            arr = np.full((160, 160, 3), base, dtype=np.uint8)
            arr[:, :, 0] = (arr[:, :, 0] + idx * 7) % 255
            arr[::4, :, 1] = 255 - arr[::4, :, 1]
            arr[:, ::5, 2] = (arr[:, ::5, 2] + 64) % 255
            Image.fromarray(arr, mode="RGB").save(out_dir / row["img_name"])

    monkeypatch.setenv("TCC_DATASET_ROOT", str(dataset_root))
    return dataset_root
