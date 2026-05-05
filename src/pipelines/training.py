from __future__ import annotations

import torch


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, y, 1.0

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    return mixed_x, y, y[perm], lam


def mixup_loss(
    criterion,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    if lam >= 1.0:
        return criterion(logits, y_a)
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
