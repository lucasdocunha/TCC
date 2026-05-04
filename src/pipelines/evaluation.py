from __future__ import annotations

from contextlib import nullcontext
import logging
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

ThresholdMetric = Literal["accuracy", "youden", "f1", "balanced_accuracy"]
ForwardFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]
LOGIT_LIMIT = 80.0


def amp_context(device: torch.device, enabled: bool = True):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def sanitize_inputs(x: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(x).all():
        return x
    logger.warning("Replacing non-finite values in evaluation inputs.")
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)


def sanitize_logits(logits: torch.Tensor, limit: float = LOGIT_LIMIT) -> torch.Tensor:
    logits = logits.float()
    if torch.isfinite(logits).all():
        return logits
    logger.warning("Replacing non-finite logits before loss/metric computation.")
    return torch.nan_to_num(
        logits,
        nan=0.0,
        posinf=limit,
        neginf=-limit,
    ).clamp(min=-limit, max=limit)


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    logits = np.nan_to_num(logits, nan=0.0, posinf=LOGIT_LIMIT, neginf=-LOGIT_LIMIT)
    probs = softmax(logits, axis=1)[:, 1]
    return clean_probabilities(probs)


def clean_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(probs, 0.0, 1.0)


def safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    probs = clean_probabilities(probs)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, probs))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def binary_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    loss: float | None = None,
    logits: np.ndarray | None = None,
    ids: np.ndarray | None = None,
) -> dict:
    y_true = np.asarray(y_true)
    probs = clean_probabilities(probs)
    threshold = float(np.nan_to_num(threshold, nan=0.5, posinf=1.0, neginf=0.0))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    y_pred = (probs >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]

    if len(y_true) == 0:
        acc = precision = recall = f1 = auc = specificity = 0.0
    else:
        acc = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        auc = safe_auc(y_true, probs)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    result = {
        "loss": float(np.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0))
        if loss is not None
        else 0.0,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "specificity": specificity,
        **counts,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }
    if logits is not None:
        result["logits"] = np.asarray(logits)
    if ids is not None:
        result["ids"] = np.asarray(ids)
    return result


def best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric: ThresholdMetric = "accuracy",
) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    probs = clean_probabilities(probs)
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], probs)))
    best_value = 0.5
    best_score = -1.0

    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(int)
        counts = confusion_counts(y_true, y_pred)
        tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if metric == "f1":
            score = float(f1_score(y_true, y_pred, zero_division=0))
        elif metric == "balanced_accuracy":
            score = (recall + specificity) / 2.0
        elif metric == "youden":
            score = recall + specificity - 1.0
        elif metric == "accuracy":
            score = float(accuracy_score(y_true, y_pred))
        else:
            raise ValueError(
                "metric must be 'accuracy', 'youden', 'f1', or 'balanced_accuracy'"
            )

        if score > best_score:
            best_score = score
            best_value = float(threshold)

    return best_value, float(best_score)


def checkpoint_score(metrics: dict) -> float:
    auc = float(metrics.get("auc", 0.0))
    return auc if auc > 0.0 else float(metrics.get("acc", 0.0))


def evaluate_classifier(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    threshold: float = 0.5,
    forward_fn: ForwardFn | None = None,
    use_amp: bool = True,
    desc: str = "Eval",
) -> dict:
    model.eval()
    losses, y_true, all_logits, all_ids = [], [], [], []

    with torch.no_grad():
        for x, y, idx in tqdm(loader, desc=desc, leave=False):
            x = sanitize_inputs(x.to(device))
            y = y.to(device)

            with amp_context(device, enabled=use_amp):
                out = forward_fn(model, x) if forward_fn is not None else model(x)
            out = sanitize_logits(out)
            loss = criterion(out, y)

            losses.append(float(torch.nan_to_num(loss.detach()).item()))
            y_true.extend(y.cpu().numpy().tolist())
            all_logits.append(out.detach().cpu().numpy().astype(np.float32))
            all_ids.extend(idx.cpu().numpy().tolist())

    if all_logits:
        logits = np.concatenate(all_logits)
        probs = probabilities_from_logits(logits)
    else:
        logits = np.empty((0, 2), dtype=np.float32)
        probs = np.empty((0,), dtype=np.float64)

    return binary_metrics(
        y_true=np.asarray(y_true),
        probs=probs,
        threshold=threshold,
        loss=float(np.mean(losses)) if losses else 0.0,
        logits=logits,
        ids=np.asarray(all_ids),
    )
