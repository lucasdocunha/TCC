import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class NonFiniteLogitModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor(
            [
                [0.0, float("nan")],
                [float("inf"), 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=x.dtype,
            device=x.device,
        )


def test_mobilenet_evaluate_sanitizes_non_finite_logits():
    from src.pipelines.mobilenet import _evaluate

    x = torch.zeros(4, 3, 16, 16)
    y = torch.tensor([0, 1, 1, 0])
    idx = torch.arange(4)
    loader = DataLoader(TensorDataset(x, y, idx), batch_size=4)
    criterion = torch.nn.CrossEntropyLoss()

    results = _evaluate(
        NonFiniteLogitModel(),
        loader,
        criterion,
        torch.device("cpu"),
    )

    assert np.isfinite(results["loss"])
    assert np.isfinite(results["probs"]).all()
    assert np.isfinite(results["auc"])


def test_shared_evaluate_classifier_sanitizes_non_finite_logits():
    from src.pipelines.evaluation import evaluate_classifier

    x = torch.zeros(4, 3, 16, 16)
    y = torch.tensor([0, 1, 1, 0])
    idx = torch.arange(4)
    loader = DataLoader(TensorDataset(x, y, idx), batch_size=4)
    criterion = torch.nn.CrossEntropyLoss()

    results = evaluate_classifier(
        NonFiniteLogitModel(),
        loader,
        criterion,
        torch.device("cpu"),
    )

    assert np.isfinite(results["loss"])
    assert np.isfinite(results["probs"]).all()
    assert np.isfinite(results["auc"])


def test_shared_metrics_return_zero_auc_for_single_class():
    from src.pipelines.evaluation import binary_metrics

    results = binary_metrics(
        y_true=np.array([1, 1, 1]),
        probs=np.array([0.2, 0.8, 0.9]),
        threshold=0.5,
    )

    assert results["auc"] == 0.0
    assert np.isfinite(results["acc"])


def test_best_threshold_ignores_non_finite_probabilities():
    from src.pipelines.evaluation import best_threshold

    threshold, score = best_threshold(
        y_true=np.array([0, 1, 0, 1]),
        probs=np.array([0.1, np.nan, np.inf, 0.9]),
        metric="balanced_accuracy",
    )

    assert np.isfinite(threshold)
    assert np.isfinite(score)


def test_binary_metrics_can_recompute_predictions_without_forward_pass():
    from src.pipelines.evaluation import binary_metrics

    y_true = np.array([0, 1, 1, 0])
    probs = np.array([0.2, 0.4, 0.7, 0.8])

    low_threshold = binary_metrics(y_true, probs, threshold=0.3)
    high_threshold = binary_metrics(y_true, probs, threshold=0.6)

    assert low_threshold["y_pred"].tolist() == [0, 1, 1, 1]
    assert high_threshold["y_pred"].tolist() == [0, 0, 1, 1]
