import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import tqdm

from src.data import ImageDataset
from src.data.paths import phase1_split_root
from src.pipelines.evaluation import (
    ThresholdMetric,
    amp_context,
    best_threshold,
    binary_metrics,
    checkpoint_score,
    evaluate_classifier,
    sanitize_logits,
)
from src.plots import plot_confusion_matrix, plot_roc_auc, save_metrics_csv

logger = logging.getLogger(__name__)


class ViTTransform:
    def __init__(self, pretrained: str, augment: bool = False):
        from transformers import ViTImageProcessor

        self.processor = ViTImageProcessor.from_pretrained(pretrained)
        self.augment = augment
        self.aug = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(
                    brightness=0.18,
                    contrast=0.18,
                    saturation=0.12,
                    hue=0.02,
                ),
                transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
            ]
        )

    def __call__(self, img):
        if self.augment:
            img = self.aug(img)
        return self.processor(images=img, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)


def collate_fn(batch):
    imgs, labels, idxs = zip(*batch)
    imgs = torch.stack([i if isinstance(i, torch.Tensor) else torch.tensor(i) for i in imgs])
    labels = torch.tensor(labels, dtype=torch.long)
    idxs = torch.tensor(idxs, dtype=torch.long)
    return imgs, labels, idxs


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split_root(_pwd: Path, _raw_min: bool, split: str) -> Path:
    return phase1_split_root(split)


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


def _vit_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(pixel_values=x).logits


def _unfreeze_vit(model: nn.Module, last_n_layers: int) -> None:
    for param in model.parameters():
        param.requires_grad = False

    last_n_layers = max(0, min(last_n_layers, len(model.vit.encoder.layer)))
    for layer in model.vit.encoder.layer[-last_n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.vit.layernorm.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def _optimizer(
    model: nn.Module,
    learning_rate_classifier: float,
    learning_rate_backbone: float,
    weight_decay: float,
    last_n_layers: int,
):
    param_groups = [
        {
            "params": model.classifier.parameters(),
            "lr": learning_rate_classifier,
            "weight_decay": weight_decay,
        },
        {
            "params": model.vit.layernorm.parameters(),
            "lr": learning_rate_backbone,
            "weight_decay": weight_decay,
        },
    ]
    layers = model.vit.encoder.layer[-last_n_layers:] if last_n_layers > 0 else []
    for offset, layer in enumerate(reversed(layers), start=1):
        param_groups.append(
            {
                "params": layer.parameters(),
                "lr": learning_rate_backbone / float(offset),
                "weight_decay": weight_decay,
            }
        )
    return torch.optim.AdamW(param_groups)


def run_vit(
    epochs: int = 50,
    raw_min: bool = False,
    data_limit: int | float | None = None,
    output_root: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pretrained_model: str = "google/vit-base-patch16-224",
    learning_rate_classifier: float = 1e-4,
    learning_rate_backbone: float = 1e-5,
    weight_decay: float = 1e-2,
    early_stop_patience: int = 12,
    last_n_layers: int = 3,
    use_weighted_sampler: bool = True,
    use_class_weights: bool = True,
    label_smoothing: float = 0.0,
    threshold_metric: ThresholdMetric = "accuracy",
    augment: bool = True,
    seed: int = 42,
    max_grad_norm: float | None = 1.0,
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
    model_name = "vit"
    run_dir = "none" if data_limit == np.inf else f"none_limit{data_limit}"
    model_dir = output_root / "models" / model_name / run_dir

    train_transform = ViTTransform(pretrained_model, augment=augment)
    eval_transform = ViTTransform(pretrained_model, augment=False)

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
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(
        train, sampler=sampler, shuffle=sampler is None, **loader_kwargs
    )
    val_loader = DataLoader(val, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test, shuffle=False, **loader_kwargs)

    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        pretrained_model,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(model.config.hidden_size, 2),
    )
    _unfreeze_vit(model, last_n_layers=last_n_layers)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=loss_weights.to(device) if use_class_weights else None,
        label_smoothing=label_smoothing,
    )
    optimizer = _optimizer(
        model,
        learning_rate_classifier=learning_rate_classifier,
        learning_rate_backbone=learning_rate_backbone,
        weight_decay=weight_decay,
        last_n_layers=last_n_layers,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    logger.info(
        "Starting ViT: model=%s train=%d val=%d test=%d classes=%s device=%s",
        pretrained_model,
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

    for epoch in tqdm.tqdm(range(1, epochs + 1), desc="Epochs"):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y, _ in tqdm.tqdm(
            train_loader, desc=f"ViT train {epoch}/{epochs}", leave=False
        ):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with amp_context(device):
                out = model(pixel_values=x).logits
            out = sanitize_logits(out)
            loss = criterion(out, y)

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
            model, val_loader, criterion, device, forward_fn=_vit_logits
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
        scheduler.step(val_metrics["auc"])

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

        current_score = checkpoint_score(val_metrics)
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
        forward_fn=_vit_logits,
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
    )
    torch.save(model.state_dict(), model_dir / "weights" / f"{model_name}.pth")

    plot_confusion_matrix(test_results, str(model_dir), f"{model_name} Confusion Matrix")
    plot_roc_auc(test_results, str(model_dir), f"{model_name} ROC-AUC Curve")
    save_metrics_csv(
        test_results,
        str(model_dir),
        extra_info={
            "model": model_name,
            "pretrained_model": pretrained_model,
            "image_size": 224,
            "epochs_requested": epochs,
            "epochs_run": epochs_run,
            "best_val_auc": best_val_auc,
            "threshold": best_threshold_value,
            "threshold_metric": threshold_metric,
            "last_n_layers": last_n_layers,
            "use_weighted_sampler": use_weighted_sampler,
            "use_class_weights": use_class_weights,
            "label_smoothing": label_smoothing,
            "learning_rate_classifier": learning_rate_classifier,
            "learning_rate_backbone": learning_rate_backbone,
            "augment": augment,
        },
    )

    logger.info(
        "ViT test: acc=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f specificity=%.4f",
        test_results["acc"],
        test_results["precision"],
        test_results["recall"],
        test_results["f1"],
        test_results["auc"],
        test_results["specificity"],
    )
    return test_results
