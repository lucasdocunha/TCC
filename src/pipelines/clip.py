from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from src.data import ImageDataset
from src.data.paths import phase1_split_root
from src.models.clip import CLIPVisionClassifier, MEAN, STD, clip_safe_name
from src.pipelines.evaluation import (
    ThresholdMetric,
    amp_context,
    best_threshold,
    binary_metrics,
    evaluate_classifier,
    sanitize_inputs,
    sanitize_logits,
)
from src.pipelines.training import mixup_batch, mixup_loss
from src.plots import plot_confusion_matrix, plot_roc_auc, save_metrics_csv

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split_root(_pwd: Path, _raw_min: bool, split: str) -> Path:
    return phase1_split_root(split)


def _transforms(image_size: int, augment: bool = True):
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.82, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.12, contrast=0.12, saturation=0.08, hue=0.015
                        )
                    ],
                    p=0.5,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    return train_transform, eval_transform


def _class_balance(
    csv_path: Path,
) -> tuple[torch.Tensor, WeightedRandomSampler, dict[int, int]]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    class_counts = df["target"].value_counts().sort_index()
    counts = {int(k): int(v) for k, v in class_counts.to_dict().items()}
    total = len(df)

    loss_weights = torch.tensor(
        [total / (2.0 * counts.get(0, 1)), total / (2.0 * counts.get(1, 1))],
        dtype=torch.float,
    )
    sample_weights = (
        df["target"].map({0: 1.0 / counts.get(0, 1), 1: 1.0 / counts.get(1, 1)}).values
    )
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(df),
        replacement=True,
    )
    return loss_weights, sampler, counts


def _clip_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def run_clip(
    epochs: int = 20,
    raw_min: bool = True,
    data_limit: int | float | None = None,
    output_root: str | Path | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 224,
    patch_size: int = 16,
    hidden_size: int = 256,
    projection_dim: int = 128,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 8,
    dropout: float = 0.2,
    train_backbone: bool = True,
    last_n_layers: int = 2,
    learning_rate_head: float = 5e-4,
    learning_rate_backbone: float = 1e-4,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 6,
    use_weighted_sampler: bool = True,
    use_class_weights: bool = True,
    label_smoothing: float = 0.0,
    threshold_metric: ThresholdMetric = "accuracy",
    augment: bool = True,
    seed: int = 42,
    max_grad_norm: float | None = 1.0,
    mixup_alpha: float = 0.0,
):
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    pwd = Path.cwd()
    output_root = Path(output_root) if output_root is not None else pwd
    data_limit = np.inf if data_limit is None else data_limit
    data_dir = pwd / "data" / ("raw_min" if raw_min else "raw")
    device = _device()
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0
    model_name = "clip"
    safe_model = clip_safe_name()
    run_dir = "none" if data_limit == np.inf else f"none_limit{data_limit}"
    model_dir = output_root / "models" / model_name / safe_model / run_dir

    train_transform, eval_transform = _transforms(image_size, augment=augment)
    train = ImageDataset(
        file_csv=data_dir / "train.csv",
        images_dir=_split_root(pwd, raw_min, "train"),
        transform=train_transform,
        data_limit=data_limit,
        fourier="none",
        spatial_size=None,
    )
    val = ImageDataset(
        file_csv=data_dir / "val.csv",
        images_dir=_split_root(pwd, raw_min, "val"),
        transform=eval_transform,
        data_limit=data_limit,
        fourier="none",
        spatial_size=None,
    )
    test = ImageDataset(
        file_csv=data_dir / "test.csv",
        images_dir=_split_root(pwd, raw_min, "test"),
        transform=eval_transform,
        data_limit=data_limit,
        fourier="none",
        spatial_size=None,
    )

    loss_weights, sampler, class_counts = _class_balance(data_dir / "train.csv")
    if data_limit != np.inf or not use_weighted_sampler:
        sampler = None

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    train_loader = DataLoader(
        train, sampler=sampler, shuffle=sampler is None, **loader_kwargs
    )
    val_loader = DataLoader(val, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test, shuffle=False, **loader_kwargs)

    model = CLIPVisionClassifier(
        num_classes=2,
        dropout=dropout,
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        projection_dim=projection_dim,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
    )
    if not train_backbone:
        model.freeze_backbone()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=loss_weights.to(device) if use_class_weights else None,
        label_smoothing=label_smoothing,
    )
    param_groups = [{"params": model.head.parameters(), "lr": learning_rate_head}]
    visual_params = [p for p in model.visual_proj.parameters() if p.requires_grad]
    if visual_params:
        param_groups.append({"params": visual_params, "lr": learning_rate_backbone})
    head_params = {id(p) for p in model.head.parameters()}
    visual_param_ids = {id(p) for p in model.visual_proj.parameters()}
    backbone_params = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in head_params and id(p) not in visual_param_ids
    ]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": learning_rate_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    logger.info(
        "Starting CLIP from scratch: train=%d val=%d test=%d classes=%s device=%s",
        len(train),
        len(val),
        len(test),
        class_counts,
        device,
    )

    best_score = -1.0
    best_val_auc = 0.0
    best_threshold_value = 0.5
    epochs_without_improvement = 0
    epochs_run = 0
    best_path = model_dir / "weights" / f"best_{model_name}.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y, _ in tqdm(
            train_loader, desc=f"CLIP train {epoch}/{epochs}", leave=False
        ):
            x = sanitize_inputs(x.to(device))
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with amp_context(device):
                train_x, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
                out = model(train_x)
            out = sanitize_logits(out)
            loss = mixup_loss(criterion, out, y_a, y_b, lam)

            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            train_correct += int((out.argmax(1) == y).sum().item())
            train_total += int(y.size(0))

        epochs_run = epoch
        val_base = evaluate_classifier(
            model, val_loader, criterion, device, forward_fn=_clip_logits
        )
        threshold, threshold_score = best_threshold(
            val_base["y_true"], val_base["probs"], metric=threshold_metric
        )
        val_metrics = binary_metrics(
            val_base["y_true"],
            val_base["probs"],
            threshold=threshold,
            loss=val_base["loss"],
            logits=val_base["logits"],
            ids=val_base["ids"],
        )
        selection_score = threshold_score
        scheduler.step(selection_score)

        train_loss /= max(len(train_loader), 1)
        train_acc = train_correct / max(train_total, 1)
        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_auc=%.4f threshold=%.4f score=%.4f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_metrics["loss"],
            val_metrics["acc"],
            val_metrics["auc"],
            threshold,
            threshold_score,
        )

        current_score = selection_score
        if current_score > best_score:
            best_score = current_score
            best_val_auc = val_metrics["auc"]
            best_threshold_value = threshold
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_results = evaluate_classifier(
        model,
        test_loader,
        criterion,
        device,
        threshold=best_threshold_value,
        forward_fn=_clip_logits,
    )

    (model_dir / "weights").mkdir(parents=True, exist_ok=True)
    (model_dir / "results").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "results" / "outputs.npz",
        logits=test_results["logits"],
        ids=test_results["ids"],
        probs=test_results["probs"],
        y_true=test_results["y_true"],
        y_pred=test_results["y_pred"],
        threshold=best_threshold_value,
    )
    torch.save(model.state_dict(), model_dir / "weights" / f"{model_name}.pth")

    plot_confusion_matrix(test_results, str(model_dir), f"{model_name} Confusion Matrix")
    plot_roc_auc(test_results, str(model_dir), f"{model_name} ROC-AUC Curve")
    save_metrics_csv(
        test_results,
        str(model_dir),
        extra_info={
            "model": model_name,
            "architecture": safe_model,
            "image_size": image_size,
            "patch_size": patch_size,
            "hidden_size": hidden_size,
            "projection_dim": projection_dim,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "epochs_requested": epochs,
            "epochs_run": epochs_run,
            "best_val_auc": best_val_auc,
            "threshold": best_threshold_value,
            "threshold_metric": threshold_metric,
            "train_backbone": train_backbone,
            "last_n_layers": last_n_layers,
            "use_weighted_sampler": use_weighted_sampler,
            "use_class_weights": use_class_weights,
            "label_smoothing": label_smoothing,
            "mixup_alpha": mixup_alpha,
            "learning_rate_head": learning_rate_head,
            "learning_rate_backbone": learning_rate_backbone,
            "augment": augment,
        },
    )

    logger.info(
        "CLIP test: acc=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f specificity=%.4f",
        test_results["acc"],
        test_results["precision"],
        test_results["recall"],
        test_results["f1"],
        test_results["auc"],
        test_results["specificity"],
    )
    return test_results
