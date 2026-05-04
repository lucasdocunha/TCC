from contextlib import nullcontext
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import tqdm

from src.data import FourierMode, ImageDataset
from src.data.paths import phase1_split_root
from src.models.resnet import freeze_backbone, resnet, unfreeze_last_blocks
from src.pipelines.evaluation import (
    binary_metrics,
    best_threshold,
    checkpoint_score,
    evaluate_classifier,
)
from src.plots import plot_confusion_matrix, plot_roc_auc, save_metrics_csv

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def _make_scaler(device: torch.device):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


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
                    image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02
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


def _class_weights(
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


def _best_accuracy_threshold(
    y_true: np.ndarray, probs: np.ndarray
) -> tuple[float, float]:
    return best_threshold(y_true, probs, metric="accuracy")


def _best_youden_threshold(
    y_true: np.ndarray, probs: np.ndarray
) -> tuple[float, float]:
    return best_threshold(y_true, probs, metric="youden")


def _evaluate(model, loader, criterion, device, threshold: float = 0.5):
    return evaluate_classifier(model, loader, criterion, device, threshold=threshold)


def run_resnet(
    epochs: int = 25,
    raw_min: bool = True,
    fourier: FourierMode = "none",
    data_limit: int | float = np.inf,
    output_root: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pretrained: bool = True,
    architecture: str = "resnet18",
    image_size: int = 224,
    train_backbone: bool = True,
    train_layer3: bool = False,
    learning_rate_head: float = 1e-3,
    learning_rate_backbone: float = 1e-4,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 8,
    use_weighted_sampler: bool = True,
    use_class_weights: bool = False,
    augment: bool = True,
    threshold_strategy: str = "accuracy",
    seed: int = 42,
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
    model_name = "resnet"
    device = _device()
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    logger.info(
        "Iniciando run_resnet: arch=%s epochs=%d image_size=%d device=%s",
        architecture,
        epochs,
        image_size,
        device,
    )

    effective_augment = augment and fourier == "none"
    train_transform, eval_transform = _transforms(image_size, augment=effective_augment)
    spatial_size = (image_size, image_size) if fourier != "none" else None
    data_dir = pwd / "data" / ("raw_min" if raw_min else "raw")

    train = ImageDataset(
        file_csv=data_dir / "train.csv",
        images_dir=_split_root(pwd, raw_min, "train"),
        transform=train_transform,
        data_limit=data_limit,
        fourier=fourier,
        spatial_size=spatial_size,
    )
    val = ImageDataset(
        file_csv=data_dir / "val.csv",
        images_dir=_split_root(pwd, raw_min, "val"),
        transform=eval_transform,
        data_limit=data_limit,
        fourier=fourier,
        spatial_size=spatial_size,
    )
    test = ImageDataset(
        file_csv=data_dir / "test.csv",
        images_dir=_split_root(pwd, raw_min, "test"),
        transform=eval_transform,
        data_limit=data_limit,
        fourier=fourier,
        spatial_size=spatial_size,
    )

    loss_weights, sampler, class_counts = _class_weights(data_dir / "train.csv")
    if data_limit != np.inf or not use_weighted_sampler:
        sampler = None

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        shuffle=sampler is None,
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    logger.info(
        "Datasets: train=%d val=%d test=%d | class_counts=%s",
        len(train),
        len(val),
        len(test),
        class_counts,
    )

    sample_x, _, _ = train[0]
    model = resnet(
        num_classes=2,
        pretrained=pretrained,
        architecture=architecture,
        in_channels=sample_x.shape[0],
    )
    freeze_backbone(model)
    if train_backbone:
        unfreeze_last_blocks(model, train_layer3=train_layer3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=loss_weights.to(device) if use_class_weights else None
    )
    param_groups = [{"params": model.fc.parameters(), "lr": learning_rate_head}]
    if train_backbone:
        param_groups.append(
            {"params": model.layer4.parameters(), "lr": learning_rate_backbone}
        )
        if train_layer3:
            param_groups.append(
                {
                    "params": model.layer3.parameters(),
                    "lr": learning_rate_backbone * 0.5,
                }
            )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )
    scaler = _make_scaler(device)

    best_score = -1.0
    best_val_auc = 0.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    model_dir = output_root / "models" / model_name
    if fourier != "none":
        model_dir = model_dir / fourier
    best_path = model_dir / "weights" / f"best_{model_name}.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm.tqdm(range(epochs), desc="Epochs"):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y, _ in tqdm.tqdm(
            train_loader, desc=f"Train ep {epoch + 1}/{epochs}", leave=False
        ):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with _amp_context(device):
                out = model(x)
                loss = criterion(out, y)

            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            train_loss += loss.item()
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += y.size(0)

        val_base = _evaluate(model, val_loader, criterion, device)
        if threshold_strategy == "accuracy":
            threshold, threshold_score = _best_accuracy_threshold(
                val_base["y_true"], val_base["probs"]
            )
        elif threshold_strategy == "youden":
            threshold, threshold_score = _best_youden_threshold(
                val_base["y_true"], val_base["probs"]
            )
        elif threshold_strategy == "fixed":
            threshold, threshold_score = 0.5, val_base["acc"]
        else:
            raise ValueError(
                "threshold_strategy must be 'accuracy', 'youden', or 'fixed'"
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
            "Época %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_auc=%.4f val_threshold=%.4f threshold_score=%.4f",
            epoch + 1,
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
            best_threshold = threshold
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
            logger.info("Novo melhor checkpoint salvo em %s", best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                logger.info("Early stopping ativado na época %d.", epoch + 1)
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_results = _evaluate(
        model, test_loader, criterion, device, threshold=best_threshold
    )

    y_true = test_results["y_true"]
    y_pred = test_results["y_pred"]
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    test_results["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    test_results.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn})

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

    plot_confusion_matrix(
        test_results, str(model_dir), f"{model_name} Confusion Matrix"
    )
    plot_roc_auc(test_results, str(model_dir), f"{model_name} ROC-AUC Curve")
    save_metrics_csv(
        test_results,
        str(model_dir),
        extra_info={
            "architecture": architecture,
            "fourier": fourier,
            "in_channels": sample_x.shape[0],
            "image_size": image_size,
            "epochs_requested": epochs,
            "best_val_auc": best_val_auc,
            "threshold": best_threshold,
            "pretrained": pretrained,
            "train_backbone": train_backbone,
            "train_layer3": train_layer3,
            "use_weighted_sampler": use_weighted_sampler,
            "use_class_weights": use_class_weights,
            "augment": effective_augment,
            "augment_requested": augment,
            "threshold_strategy": threshold_strategy,
        },
    )

    logger.info(
        "Métricas no teste: acc=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f specificity=%.4f | tp=%d fp=%d fn=%d tn=%d",
        test_results["acc"],
        test_results["precision"],
        test_results["recall"],
        test_results["f1"],
        test_results["auc"],
        test_results["specificity"],
        tp,
        fp,
        fn,
        tn,
    )
    return test_results
