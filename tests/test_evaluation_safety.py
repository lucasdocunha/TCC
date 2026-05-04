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
