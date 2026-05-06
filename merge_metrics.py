import pandas as pd
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parent / "models"
OUTPUT_FILE = MODELS_DIR / "all_metrics_v3.csv"


def _extract_path_info(csv_path: Path) -> dict[str, str]:
    parts = csv_path.relative_to(MODELS_DIR).parts  # ex: ("resnet", "magnitude", "results", "metrics_summary.csv")

    hierarchy = parts[:-2]

    return {"source_path": str(Path(*hierarchy))}


def merge_all_metrics() -> pd.DataFrame:
    csv_files = sorted(MODELS_DIR.rglob("results/metrics_summary.csv"))

    if not csv_files:
        print(f"Nenhum metrics_summary.csv encontrado em {MODELS_DIR}")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        path_info = _extract_path_info(csv_path)
        df.insert(0, "source_path", path_info["source_path"])
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged


def main() -> None:
    merged = merge_all_metrics()

    if merged.empty:
        return

    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"{len(merged)} runs consolidados em: {OUTPUT_FILE}")
    print(merged.to_string(index=False, max_cols=10))


if __name__ == "__main__":
    main()
