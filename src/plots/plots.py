import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)

from src.pipelines.evaluation import clean_probabilities, probabilities_from_logits


def plot_confusion_matrix(
    test_results: dict,
    model_dir: str,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple = (7, 6),
    dpi: int = 150,
):
    """
    Plota a matriz de confusão com métricas adicionais (acc, precision, recall, f1).

    Args:
        test_results : dict com chaves 'y_true' e 'y_pred'
        model_dir    : raiz do modelo (ex: models/meu_modelo)
        title        : título do plot
        cmap         : colormap do seaborn
        figsize      : tamanho da figura
        dpi          : resolução de saída
    """
    y_true = test_results["y_true"]
    y_pred = test_results["y_pred"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ── labels com contagem absoluta ──────────────────────────────────────────
    labels = np.array([
        [f"{v}\n({p:.1%})" for v, p in zip(row_c, row_n)]
        for row_c, row_n in zip(cm, cm_norm)
    ])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.heatmap(
        cm_norm,
        annot=labels,
        fmt="",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        vmin=0, vmax=1,
        cbar_kws={"label": "Row-normalized rate"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xticklabels(["Negative (0)", "Positive (1)"], fontsize=10)
    ax.set_yticklabels(["Negative (0)", "Positive (1)"], fontsize=10, va="center")

    # ── caixa de métricas ─────────────────────────────────────────────────────
    """metrics_text = (
        f"Accuracy   : {acc:.4f}\n"
        f"Precision  : {precision:.4f}\n"
        f"Recall     : {recall:.4f}\n"
        f"Specificity: {specificity:.4f}\n"
        f"F1-Score   : {f1:.4f}\n"
        f"\nTP={tp}  FP={fp}\nFN={fn}  TN={tn}"
    )
    fig.text(
        1.01, 0.5, metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        fontfamily="monospace",
    )
    """
    plt.tight_layout()

    save_dir = os.path.join(model_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Confusion matrix salva em: {out_path}")


def plot_roc_auc(
    test_results: dict,
    model_dir: str,
    title: str = "ROC-AUC Curve",
    color: str = "#4C72B0",
    figsize: tuple = (7, 6),
    dpi: int = 150,
):
    """
    Plota a curva ROC com AUC destacado.

    Args:
        test_results : dict com chaves 'y_true' e 'probs' (probabilidades da classe positiva)
        model_dir    : raiz do modelo
        title        : título do plot
        color        : cor da curva ROC
        figsize      : tamanho da figura
        dpi          : resolução de saída
    """
    y_true = test_results["y_true"]

    # aceita tanto 'probs' quanto 'logits' como fallback
    if "probs" in test_results:
        scores = clean_probabilities(np.array(test_results["probs"]))
    else:
        logits = np.array(test_results["logits"])
        scores = probabilities_from_logits(logits) if logits.ndim == 2 else logits

    # se vier shape (N, 2), pega coluna 1
    if scores.ndim == 2:
        scores = scores[:, 1]
    scores = clean_probabilities(scores)

    if len(np.unique(y_true)) < 2:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        thresholds = np.array([0.5, 0.5])
        auc = 0.0
    else:
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)

    # ponto de maior Youden Index (sensibilidade + especificidade - 1)
    youden_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[youden_idx]
    best_fpr, best_tpr = fpr[youden_idx], tpr[youden_idx]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"AUC = {auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color=color)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier")

    # marca o melhor threshold
    ax.scatter(
        best_fpr, best_tpr,
        color="crimson", zorder=5, s=80,
        label=f"Best threshold = {best_thresh:.3f}\n(FPR={best_fpr:.3f}, TPR={best_tpr:.3f})",
    )
    ax.axvline(best_fpr, color="crimson", linestyle=":", lw=1, alpha=0.5)
    ax.axhline(best_tpr, color="crimson", linestyle=":", lw=1, alpha=0.5)

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    sns.despine(ax=ax)
    plt.tight_layout()

    save_dir = os.path.join(model_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "roc_auc.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] ROC-AUC curve salva em: {out_path}")


def save_metrics_csv(
    test_results: dict,
    model_dir: str,
    extra_info: dict = None,
):
    """
    Salva um CSV com todas as métricas escalares + predictions por amostra.

    Gera dois arquivos:
        results/metrics_summary.csv  → uma linha por run (métricas escalares)
        results/predictions.csv      → uma linha por amostra (id, y_true, y_pred, prob)

    Args:
        test_results : dict completo
        model_dir    : raiz do modelo
        extra_info   : dict opcional com metadados extras (ex: {'model': 'bert', 'epoch': 5})
    """
    save_dir = os.path.join(model_dir, "results")
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. métricas escalares ─────────────────────────────────────────────────
    scalar_keys = ["acc", "precision", "f1", "auc", "recall", "specificity"]
    summary = {k: test_results[k] for k in scalar_keys if k in test_results}

    # recalcula recall e specificity se não estiverem no dict
    y_true = np.array(test_results["y_true"])
    y_pred = np.array(test_results["y_pred"])
    if "recall" not in summary:
        summary["recall"] = recall_score(y_true, y_pred, zero_division=0)
    if "specificity" not in summary:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        summary["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        summary["tp"], summary["fp"] = int(tp), int(fp)
        summary["fn"], summary["tn"] = int(fn), int(tn)

    if extra_info:
        summary.update(extra_info)

    pd.DataFrame([summary]).to_csv(
        os.path.join(save_dir, "metrics_summary.csv"), index=False
    )
    print(f"[✓] metrics_summary.csv salvo em: {save_dir}")

    # ── 2. predictions por amostra ────────────────────────────────────────────
    pred_df = pd.DataFrame({"id": test_results["ids"], "y_true": y_true, "y_pred": y_pred})

    if "probs" in test_results:
        probs = np.array(test_results["probs"])
    else:
        logits = np.array(test_results["logits"])
        probs = probabilities_from_logits(logits) if logits.ndim == 2 else logits

    if probs.ndim == 2:
        pred_df["prob_neg"] = clean_probabilities(probs[:, 0])
        pred_df["prob_pos"] = clean_probabilities(probs[:, 1])
    else:
        pred_df["prob_pos"] = clean_probabilities(probs)

    pred_df["correct"] = (pred_df["y_true"] == pred_df["y_pred"]).astype(int)

    pred_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
    print(f"[✓] predictions.csv salvo em: {save_dir}")
