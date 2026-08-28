"""Losses for reward-model training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bt_loss(s_pos: torch.Tensor, s_neg: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry pairwise ranking loss: -log sigmoid(s_pos - s_neg)."""
    return F.softplus(s_neg - s_pos).mean()


def grade_loss(logit: torch.Tensor, grade: torch.Tensor) -> torch.Tensor:
    """Auxiliary absolute-grade regression (enable once T1 labels exist).

    grade in 1..5 -> target score in (0,1): (g-1)/4, matched against sigmoid(logit).
    """
    target = (grade.float() - 1.0) / 4.0
    return F.mse_loss(torch.sigmoid(logit), target)
