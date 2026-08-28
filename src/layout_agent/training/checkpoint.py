"""Shared checkpoint helpers for DDP-trained reward models."""

from __future__ import annotations

from typing import Dict, Mapping

import torch


def strip_ddp_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip ``module.`` prefixes written by DistributedDataParallel."""
    if not any(k.startswith("module.") for k in state_dict):
        return dict(state_dict)
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}
