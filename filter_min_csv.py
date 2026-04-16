import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
IMG_BASE = Path("min_dataset")
OUT_DIR = Path("data/raw_min")

OUT_DIR.mkdir(parents=True, exist_ok=True)

splits = ["train", "val", "test"]

for split in splits:
    csv_path = RAW_DIR / f"{split}.csv"
    img_dir = IMG_BASE / split

    df = pd.read_csv(csv_path)

    images_in_folder = set(p.name for p in img_dir.iterdir() if p.is_file())

    before = len(df)
    df_filtered = df[df["img_name"].isin(images_in_folder)].reset_index(drop=True)
    after = len(df_filtered)

    out_path = OUT_DIR / f"{split}.csv"
    df_filtered.to_csv(out_path, index=False)

    print(f"[{split}] {before} → {after} linhas | salvo em {out_path}")
