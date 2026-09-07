# Benchmarking Spatial, Spectral, and Self-Supervised Cues for Face Forgery Detection under Realistic Degradation

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-50%20passed-brightgreen.svg)]()
[![Conference](https://img.shields.io/badge/SIBGRAPI-2026-darkblue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official implementation and reproducibility benchmark for the paper:
> **Benchmarking Spatial, Spectral, and Self-Supervised Cues for Face Forgery Detection under Realistic Degradation**  
> *Conference on Graphics, Patterns and Images (SIBGRAPI 2026)*

---

## Abstract & Scientific Overview

Face forgery detection models often demonstrate near-perfect accuracy on clean, uncompressed benchmarks but suffer catastrophic performance drops when deployed in real-world scenarios plagued by social media compression, noise, resizing, and blur.

This repository provides a unified, rigorous, and fully reproducible benchmark evaluating:
- **6 Model Families**:
  - Convolutional Neural Networks: **ResNet-18**, **Xception**, **MobileNetV3-Large**.
  - Vision Transformers: **ViT-B/16** (ImageNet-21k), **CLIP ViT-B/16** (OpenAI multimodal).
  - Self-Supervised Representation: **DINOv3** (ConvNeXt-Tiny backbone).
- **7 Spatial and 2D-FFT Spectral Representations**:
  - `none`: Spatial domain standard RGB (3 channels).
  - `magnitude`: FFT Log-Magnitude $\log(|F| + 1)$ (1 channel).
  - `phase`: FFT Phase angle $[0, 1]$ (1 channel).
  - `complex`: Real and Imaginary components (2 channels).
  - `concat`: Spatial RGB concatenated with FFT Magnitude (4 channels).
  - `frequency_3`: High-pass filtered magnitude representation (3 channels).
  - `concat_frequency`: Full multi-domain representation combining RGB, Magnitude, Phase, and High-Pass (6 channels).
- **2 Training Regimes**:
  - **Scratch**: Training with random initialization across all channel formats.
  - **Fine-Tuning**: Transfer learning with pre-trained backbones adapted to 1, 2, 3, 4, and 6 input channels via custom first-layer weight projections.
- **Robustness under Degradation**:
  - Multi-split evaluation evaluating generalization on clean validation/test sets (`val`, `test`) vs. degraded in-the-wild partitions (`test_d`).
- **Ensemble Strategies & Explainability (XAI)**:
  - Parallel exhaustive subset search and greedy fusion over multi-seed candidates.
  - Attribution heatmaps using layer-hook **Grad-CAM** (CNNs/ConvNeXt) and residual **Attention Rollout** (Vision Transformers).

---

## Repository Architecture

```text
├── configs/                  # Model and training YAML configurations
│   ├── base.yaml             # Shared hyperparameters (epochs, scheduler, AMP, seeds)
│   ├── clip.yaml             # CLIP ViT-B/16 configuration
│   ├── dino.yaml             # DINOv3 ConvNeXt configuration
│   ├── mobilenet.yaml        # MobileNetV3 configuration
│   ├── resnet.yaml           # ResNet-18 configuration
│   ├── vit.yaml              # ViT-B/16 configuration
│   └── xception.yaml         # Xception configuration
├── data/                     # Dataset metadata and CSV split manifests
│   ├── raw/                  # Full dataset split manifests (train.csv, val.csv, test.csv)
│   └── raw_min/              # Minimal dataset split manifests for smoke tests
├── figures/                  # Publication figures, plots, and attribution heatmaps
│   ├── architecture.png      # Pipeline architecture diagram
│   ├── fourier/              # Spectral decomposition visualizations & PDF figures
│   │   ├── samples/          # Sample Fourier representations per class
│   │   ├── fourier_modes_comparison.png
│   │   └── fourier_transformacoes.pdf
│   └── heatmaps/             # XAI attribution maps (Grad-CAM & Attention Rollout)
│       ├── individual/       # Individual model heatmaps
│       └── same_image/       # Cross-model comparisons on identical reference faces
├── results/                  # Benchmarking results and generated tables
│   ├── ensemble/             # Multi-model ensemble evaluation reports
│   └── tables/               # Formatted LaTeX (results_paper.tex), Markdown, and CSV tables
├── src/                      # Core modular framework
│   ├── data/                 # Dataset loader, 2D FFT transforms, path resolvers
│   ├── models/               # Model definitions, first-layer channel adapters, registry
│   ├── pipelines/            # Training loop, AMP, evaluation, checkpointing, ensembles
│   ├── plots/                # Plotting utilities (ROC-AUC, confusion matrix, heatmaps)
│   └── utils/                # Multi-GPU worker scheduler
├── tests/                    # 50 unit and integration tests (100% passing)
├── train.py                  # CLI entry point for single model training
├── run_matrix.py             # CLI entry point for executing the full 126-run matrix
├── evaluate.py               # CLI entry point for checkpoint evaluation
├── ensemble.py               # CLI entry point for multi-model ensemble search and fusion
├── make_tables.py            # CLI entry point for generating LaTeX paper tables
├── generate_heatmaps.py      # CLI entry point for XAI attribution generation
├── plot_frequencia.py        # Fourier spectral visualization generator
├── plot_transformacoes.py    # Fourier transformation figures generator
└── save_fourier_samples.py   # Fourier sample image exporter
```

---

## Environment Setup

### 1. Requirements

- Python 3.11 or 3.12
- Linux (Ubuntu 20.04+ recommended)
- CUDA 12.1+ (or compatible GPU driver)

### 2. Installation via `uv` (Recommended)

Using [`uv`](https://github.com/astral-sh/uv) ensures exact dependency synchronization and PyTorch CUDA wheel resolution:

```bash
# Clone the repository
git clone https://github.com/lucasdocunha/tcc.git
cd tcc

# Install dependencies and sync virtualenv
uv sync

# Run the complete test suite (50 tests)
uv run pytest -q
```

### 3. Installation via `pip` (Alternative)

If using standard `pip`, pass the PyTorch index URL explicitly:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

> [!TIP]
> **GPU Compatibility**: The pinned `cu121` build supports GPUs up to Compute Capability 9.0 (Ada/Hopper/Ampere/Turing). For newer Blackwell architecture GPUs (`compute_cap` $\ge$ 12.0), update the PyTorch index in `pyproject.toml` to `cu128` (PyTorch $\ge$ 2.9) and execute `uv lock`.

---

## Data Preparation & Environment Variables

The pipeline supports both local directories and cluster environments via environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `TCC_DATASET_ROOT` | Root directory containing image folders (`trainset/`, `valset/`, `testset/`) | `/media/ssd2/lucas.ocunha/datasets/phase1` |
| `TCC_DATA_ROOT` | Root directory containing CSV split manifests (`raw/` and `raw_min/`) | `<repo>/data` |
| `TCC_MODELS_ROOT` | Directory where trained checkpoints and metrics are saved | `<repo>/models` |
| `TCC_OUTPUT_ROOT` | Root directory for outputs (`figures/` and `results/`) | `<repo>` |

You can export these variables in your shell or `.bashrc`:
```bash
export TCC_DATASET_ROOT="/path/to/datasets/phase1"
export TCC_DATA_ROOT="/path/to/tcc/data"
export TCC_MODELS_ROOT="/path/to/tcc/models"
export TCC_OUTPUT_ROOT="/path/to/tcc"
```

---

## Step-by-Step Reproduction Guide

### Step 1: Smoke Testing (Local Verification)

Verify the full pipeline in seconds on CPU or local GPU using a minimal subset:

```bash
# Train a ResNet-18 on raw_min for 1 epoch on 32 images
uv run python train.py --config configs/resnet.yaml --fourier none --regime scratch --seed 42 --epochs 1 --data-limit 32 --raw-min
```

### Step 2: Training Individual Models

Train any architecture with any Fourier mode and regime:

```bash
# Example 1: ResNet-18 with concatenated RGB+FFT Magnitude from scratch
uv run python train.py --config configs/resnet.yaml --fourier concat --regime scratch --seed 42

# Example 2: Pre-trained ViT-B/16 with FFT Magnitude fine-tuning
uv run python train.py --config configs/vit.yaml --fourier magnitude --regime finetune --seed 42

# Example 3: Pre-trained DINOv3 (ConvNeXt) on Spatial RGB
uv run python train.py --config configs/dino.yaml --fourier none --regime finetune --seed 42
```

Trained checkpoints and evaluation artifacts are organized systematically:
```text
models/<family>/<fourier_mode>/<regime>/seed_<seed>/
├── weights/
│   ├── best.pth
│   └── final.pth
├── results/
│   ├── run_config.json
│   ├── metrics_val.csv
│   ├── outputs_val.npz
│   └── predictions_val.csv
└── plots/
    ├── confusion_matrix.png
    └── roc_auc.png
```

### Step 3: Executing the Full Benchmark Matrix

The complete paper benchmark encompasses **126 models** (6 families × 7 Fourier modes × 3 random seeds: `42`, `123`, `2024`). Use `run_matrix.py` to distribute runs across available GPUs:

```bash
# Dry run to inspect the scheduled 126 runs
uv run python run_matrix.py --regime scratch --dry-run

# Run full fine-tuning matrix distributed over GPUs 0, 1, 2, 3
uv run python run_matrix.py --regime finetune --gpus 0,1,2,3 --workers-per-gpu 1

# Run only ResNet and DINOv3 on GPU 0
uv run python run_matrix.py --regime finetune --only resnet,dino --gpus 0
```

### Step 4: Model Evaluation on Clean and Degraded Splits

Evaluate all trained checkpoints across clean validation/test sets and degraded test sets (`test_d`):

```bash
# Standard evaluation on val and test splits
uv run python evaluate.py --splits val,test

# Extended evaluation including degraded in-the-wild partition (test_d)
uv run python evaluate.py \
  --splits val,test,test_d \
  --test-d-csv /path/to/test_d.csv \
  --test-d-images-dir /path/to/test_d/images
```

This aggregates all metrics into `models/all_metrics_by_split.csv`.

### Step 5: Multi-Model Ensembles

Evaluate multi-model fusion and ensemble selection:

```bash
# Parallel exhaustive subset search across the best validation mode per family (at most 12 candidates)
uv run python ensemble.py --strategy search --pool best-mode

# Greedy forward selection across all 84 family x mode combinations
uv run python ensemble.py --strategy search --pool all

# Weighted average by validation AUC
uv run python ensemble.py --strategy weighted --pool best-mode
```

Ensemble results and selected candidate weights are saved to `results/ensemble/`.

### Step 6: Generating Paper LaTeX Tables

Generate publication-ready booktabs LaTeX tables (`results_paper.tex`), formatted Markdown (`results_full.md`), and raw CSVs:

```bash
uv run python make_tables.py
```

Outputs are saved directly to `results/tables/`:
- `results_paper.tex`: Contains formatted `mean ± std` LaTeX tables for each evaluated split (`val`, `test`, `test_d`).
- `results_full.csv`: Full tabular metric summary.
- `results_full.md`: Markdown summary.

### Step 7: Explainability & Heatmap Generation (XAI)

Generate attribution heatmaps to inspect whether models focus on facial artifacts, frequency anomalies, or background cues:

```bash
# Generate single image heatmap (Auto: Grad-CAM for CNNs/DINOv3, Attention Rollout for ViT/CLIP)
uv run python generate_heatmaps.py \
  --checkpoint models/resnet/none/scratch/seed_42/weights/best.pth \
  --image /path/to/sample.jpg

# Generate a grid comparison across multiple images
uv run python generate_heatmaps.py \
  --checkpoint models/vit/none/finetune/seed_42/weights/best.pth \
  --image img1.jpg img2.jpg img3.jpg \
  --grid \
  --output figures/heatmaps/vit_grid.png
```

To reproduce paper Fourier decomposition figures:
```bash
uv run python plot_frequencia.py
uv run python plot_transformacoes.py
uv run python save_fourier_samples.py
```

---

## Summary of Main Findings

1. **Clean vs. Degraded Generalization Gap**:
   - Classical detectors like Xception achieve high ROC-AUC on clean data ($\approx 0.884$) but suffer severe performance collapse under realistic degradation ($\approx 0.582$).
2. **Self-Supervised Representation Superiority**:
   - Pre-trained self-supervised backbones (**DINOv3**) preserve high discriminability under domain shift and compression artifacts compared to purely supervised baselines.
3. **Spectral vs. Spatial Complementarity**:
   - Purely spectral representations (Phase, Magnitude) alone underperform full RGB inputs, but hybrid representations (`concat`: Spatial + FFT Magnitude) provide enhanced robustness when spatial cues are degraded.
4. **Attribution Analysis**:
   - Grad-CAM and Attention Rollout reveal that transformer and self-supervised models maintain attention on facial landmarks, whereas standard CNNs overfit to high-frequency background noise.

---

## Test Suite

The project includes an extensive suite of **50 automated tests** covering data loaders, 2D FFT transformations, first-layer channel adaptation, architecture smoke tests, checkpoint safety, ensemble strategies, table generation, and end-to-end workflows:

```bash
uv run pytest -v
```

---

## Authors & Contributors

*Pontifical Catholic University of Paraná (PUCPR), Curitiba, PR, Brazil*

| Author | Email | GitHub | LinkedIn |
| :--- | :--- | :---: | :---: |
| **Lucas Cunha** | [`lucas.ocunha@ppgia.pucpr.br`](mailto:lucas.ocunha@ppgia.pucpr.br) | [![GitHub](https://img.shields.io/badge/GitHub-lucasdocunha-181717?logo=github)](https://github.com/lucasdocunha) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-lucasdoc-0A66C2?logo=linkedin)](https://www.linkedin.com/in/lucasdoc) |
| **Lucas Sotomaior** | [`lucas.apereira@ppgia.pucpr.br`](mailto:lucas.apereira@ppgia.pucpr.br) | [![GitHub](https://img.shields.io/badge/GitHub-LucasSotomaiorAPereira-181717?logo=github)](https://github.com/LucasSotomaiorAPereira) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-lucassotomaior-0A66C2?logo=linkedin)](https://www.linkedin.com/in/lucassotomaior/) |
| **Lucas Gasperin** | [`lucas.gasperin@pucpr.edu.br`](mailto:lucas.gasperin@pucpr.edu.br) | [![GitHub](https://img.shields.io/badge/GitHub-Lucas--PG-181717?logo=github)](https://github.com/Lucas-PG) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-lucas--gasperin-0A66C2?logo=linkedin)](https://www.linkedin.com/in/lucas-gasperin/) |
| **Beatriz Caldas** | [`beatriz.caldas@pucpr.edu.br`](mailto:beatriz.caldas@pucpr.edu.br) | [![GitHub](https://img.shields.io/badge/GitHub-beatriz--caldas-181717?logo=github)](https://github.com/beatriz-caldas) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-beatrizcaldas01-0A66C2?logo=linkedin)](https://www.linkedin.com/in/beatrizcaldas01/) |
| **Eduardo Pianovski** | [`eduardo.pianovski@pucpr.edu.br`](mailto:eduardo.pianovski@pucpr.edu.br) | [![GitHub](https://img.shields.io/badge/GitHub-dudupnetto-181717?logo=github)](https://github.com/dudupnetto) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-eduardo--pianovski-0A66C2?logo=linkedin)](https://www.linkedin.com/in/eduardo-pianovski-netto-357320269/) |
| **Rayson Laroca** | [`rayson@ppgia.pucpr.br`](mailto:rayson@ppgia.pucpr.br) | [![GitHub](https://img.shields.io/badge/GitHub-raysonlaroca-181717?logo=github)](https://github.com/raysonlaroca) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-raysonlaroca-0A66C2?logo=linkedin)](https://www.linkedin.com/in/raysonlaroca/) |



---

## Citation

If you use this codebase or benchmark in your research, please cite our paper:

```bibtex
@article{cunha2026benchmarking,
  title = {Benchmarking Spatial, Spectral, and Self-Supervised Cues for Face Forgery Detection under Realistic Degradation},
  author = {Lucas {Cunha} and Lucas {Sotomaior} and Lucas {Gasperin} and Beatriz {Caldas} and Eduardo {Pianovski} and Rayson {Laroca}},
  year = {2026},
  journal = {Conference on Graphics, Patterns and Images (SIBGRAPI)},
  volume = {},
  number = {},
  pages = {1-6},
  doi = {},
  issn = {},
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


