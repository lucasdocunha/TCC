import numpy as np


def test_plots_and_metric_export_handle_single_class_outputs(tmp_path):
    from src.plots import plot_confusion_matrix, plot_roc_auc, save_metrics_csv

    results = {
        "acc": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "auc": 0.0,
        "y_true": np.array([1, 1]),
        "y_pred": np.array([1, 1]),
        "probs": np.array([0.7, np.nan]),
        "ids": np.array([3, 4]),
    }

    plot_confusion_matrix(results, str(tmp_path))
    plot_roc_auc(results, str(tmp_path))
    save_metrics_csv(results, str(tmp_path))

    assert (tmp_path / "plots" / "confusion_matrix.png").exists()
    assert (tmp_path / "plots" / "roc_auc.png").exists()
    assert (tmp_path / "results" / "metrics_summary.csv").exists()
    assert (tmp_path / "results" / "predictions.csv").exists()
