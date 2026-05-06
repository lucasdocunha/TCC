from contextlib import nullcontext
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from src.data import ALL_FOURIER_MODES, FourierMode, ImageDataset
from src.data.paths import phase1_split_root
from src.models.mobilenet import freeze_classifier_only, mobilenet, unfreeze_last_blocks
from src.pipelines.evaluation import (
    ThresholdMetric,
    best_threshold,
    binary_metrics,
    evaluate_classifier,
    safe_auc,
)
from src.pipelines.training import mixup_batch, mixup_loss
from src.plots import plot_confusion_matrix, plot_roc_auc, save_metrics_csv

logger = logging.getLogger(__name__)
MobileNetVariant = Literal["small", "large"]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def _split_root(_pwd: Path, _raw_min: bool, split: str) -> Path:
    """Imagens sempre em phase1/{trainset,valset,testset}; só o CSV muda (raw_min vs raw)."""
    return phase1_split_root(split)


def _transforms(image_size: int, augment: bool = True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
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
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
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


def _safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    return safe_auc(y_true, probs)


def _evaluate(model, loader, criterion, device, threshold: float = 0.5):
    return evaluate_classifier(model, loader, criterion, device, threshold=threshold)


def _best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric: ThresholdMetric = "accuracy",
) -> tuple[float, float]:
    return best_threshold(y_true, probs, metric=metric)


def run_mobilenet(
    input_mode: FourierMode = "none",
    variant: MobileNetVariant = "small",
    epochs: int = 20,
    raw_min: bool = True,
    data_limit: int | float | None = None,
    output_root: str | Path | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pretrained: bool = False,
    image_size: int = 224,
    learning_rate_classifier: float = 1e-3,
    learning_rate_backbone: float | None = None,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 5,
    last_n_blocks: int | None = None,
    warmup_epochs: int = 1,
    use_weighted_sampler: bool = True,
    use_class_weights: bool = False,
    label_smoothing: float = 0.0,
    dropout: float | None = None,
    mixup_alpha: float = 0.0,
    threshold_metric: Literal[
        "accuracy", "youden", "f1", "balanced_accuracy"
    ] = "accuracy",
    augment: bool = True,
    seed: int = 42,
):
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    if variant not in ("small", "large"):
        raise ValueError("variant must be 'small' or 'large'")

    torch.manual_seed(seed)
    np.random.seed(seed)

    pwd = Path.cwd()
    output_root = Path(output_root) if output_root is not None else pwd
    data_limit = np.inf if data_limit is None else data_limit
    data_dir = pwd / "data" / ("raw_min" if raw_min else "raw")
    device = _device()
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0
    model_name = f"mobilenetv3_{variant}"
    run_dir = input_mode if data_limit == np.inf else f"{input_mode}_limit{data_limit}"
    model_dir = output_root / "models" / "mobilenet" / model_name / run_dir

    if learning_rate_backbone is None:
        learning_rate_backbone = 5e-5 if variant == "small" else 2e-5
    if last_n_blocks is None:
        last_n_blocks = 4 if variant == "small" else 3

    effective_augment = augment and input_mode == "none"
    train_transform, eval_transform = _transforms(image_size, augment=effective_augment)
    spatial_size = (image_size, image_size) if input_mode != "none" else None

    train = ImageDataset(
        file_csv=data_dir / "train.csv",
        images_dir=_split_root(pwd, raw_min, "train"),
        transform=train_transform,
        data_limit=data_limit,
        fourier=input_mode,
        spatial_size=spatial_size,
    )
    val = ImageDataset(
        file_csv=data_dir / "val.csv",
        images_dir=_split_root(pwd, raw_min, "val"),
        transform=eval_transform,
        data_limit=data_limit,
        fourier=input_mode,
        spatial_size=spatial_size,
    )
    test = ImageDataset(
        file_csv=data_dir / "test.csv",
        images_dir=_split_root(pwd, raw_min, "test"),
        transform=eval_transform,
        data_limit=data_limit,
        fourier=input_mode,
        spatial_size=spatial_size,
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

    sample_x, _, _ = train[0]
    model = mobilenet(
        num_classes=2,
        in_channels=sample_x.shape[0],
        pretrained=pretrained,
        variant=variant,
        dropout=dropout,
    )
    if pretrained:
        freeze_classifier_only(model)
    if pretrained and warmup_epochs <= 0:
        unfreeze_last_blocks(model, last_n_blocks=last_n_blocks)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=loss_weights.to(device) if use_class_weights else None,
        label_smoothing=label_smoothing,
    )

    def make_optimizer():
        param_groups = [
            {"params": model.classifier.parameters(), "lr": learning_rate_classifier}
        ]
        if pretrained:
            backbone_params = [
                p
                for p in model.features[-last_n_blocks:].parameters()
                if p.requires_grad
            ]
        else:
            classifier_params = {id(p) for p in model.classifier.parameters()}
            backbone_params = [
                p
                for p in model.parameters()
                if p.requires_grad and id(p) not in classifier_params
            ]
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": learning_rate_backbone}
            )
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    optimizer = make_optimizer()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    logger.info(
        "Starting MobileNet: variant=%s mode=%s channels=%d train=%d val=%d test=%d classes=%s device=%s",
        variant,
        input_mode,
        sample_x.shape[0],
        len(train),
        len(val),
        len(test),
        class_counts,
        device,
    )

    best_score = -1.0
    best_val_auc = 0.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    best_path = model_dir / "weights" / f"best_{model_name}.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        if pretrained and warmup_epochs > 0 and epoch == warmup_epochs + 1:
            unfreeze_last_blocks(model, last_n_blocks=last_n_blocks)
            optimizer = make_optimizer()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y, _ in tqdm(
            train_loader, desc=f"MobileNet train {epoch}/{epochs}", leave=False
        ):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with _amp_context(device):
                train_x, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
                out = model(train_x)
                loss = mixup_loss(criterion, out, y_a, y_b, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            train_correct += int((out.argmax(1) == y).sum().item())
            train_total += int(y.size(0))

        epochs_run = epoch
        val_base = _evaluate(model, val_loader, criterion, device)
        threshold, threshold_score = _best_threshold(
            val_base["y_true"],
            val_base["probs"],
            metric=threshold_metric,
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
            best_threshold = threshold
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_results = _evaluate(
        model, test_loader, criterion, device, threshold=best_threshold
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
        threshold=best_threshold,
    )
    torch.save(model.state_dict(), model_dir / "weights" / f"{model_name}.pth")

    plot_confusion_matrix(
        test_results, str(model_dir), f"{model_name} Confusion Matrix"
    )
    plot_roc_auc(test_results, str(model_dir), f"{model_name} ROC-AUC Curve")
    save_metrics_csv(
        test_results,
        str(model_dir),
        extra_info={
            "model": model_name,
            "variant": variant,
            "input_mode": input_mode,
            "in_channels": sample_x.shape[0],
            "image_size": image_size,
            "epochs_requested": epochs,
            "epochs_run": epochs_run,
            "best_val_auc": best_val_auc,
            "threshold": best_threshold,
            "threshold_metric": threshold_metric,
            "pretrained": pretrained,
            "last_n_blocks": last_n_blocks,
            "warmup_epochs": warmup_epochs,
            "use_weighted_sampler": use_weighted_sampler,
            "use_class_weights": use_class_weights,
            "label_smoothing": label_smoothing,
            "dropout": dropout,
            "mixup_alpha": mixup_alpha,
            "learning_rate_classifier": learning_rate_classifier,
            "learning_rate_backbone": learning_rate_backbone,
            "augment": effective_augment,
            "augment_requested": augment,
        },
    )

    logger.info(
        "MobileNet test: acc=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f specificity=%.4f",
        test_results["acc"],
        test_results["precision"],
        test_results["recall"],
        test_results["f1"],
        test_results["auc"],
        test_results["specificity"],
    )
    return test_results


def run_all_mobilenet_modes(**kwargs):
    return {
        mode: run_mobilenet(input_mode=mode, **kwargs) for mode in ALL_FOURIER_MODES
    }
