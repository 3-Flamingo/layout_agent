"""Candidate-scoring actor and global-state critic."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from layout_agent.training.rl_features import ACTION_FEATURE_DIM, GLOBAL_FEATURE_DIM


class CandidatePolicyNet(nn.Module):
    """Score each variable action from global image/vector context."""

    def __init__(
        self,
        in_channels: int = 3,
        global_dim: int = GLOBAL_FEATURE_DIM,
        action_dim: int = ACTION_FEATURE_DIM,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.config = {
            "in_channels": int(in_channels),
            "global_dim": int(global_dim),
            "action_dim": int(action_dim),
            "hidden_dim": int(hidden_dim),
        }
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.context = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.candidate = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.action_bias = nn.Linear(hidden_dim, 1)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
        global_features: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_context = self.image_encoder(image)
        vector_context = self.global_encoder(global_features)
        context = self.context(torch.cat((image_context, vector_context), dim=-1))
        candidates = self.candidate(action_features)
        query = self.query(context).unsqueeze(1)
        logits = (candidates * query).sum(dim=-1) / math.sqrt(candidates.shape[-1])
        logits = logits + self.action_bias(candidates).squeeze(-1)
        mask = action_mask.bool()
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("every observation must expose at least one legal action")
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        value = self.critic(context).squeeze(-1)
        return logits, value

    def distribution(
        self,
        observation: Dict[str, torch.Tensor],
    ) -> Tuple[Categorical, torch.Tensor]:
        logits, value = self(
            observation["image"],
            observation["global_features"],
            observation["action_features"],
            observation["action_mask"],
        )
        return Categorical(logits=logits), value

    def act(
        self,
        observation: Dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution(observation)
        action = (
            distribution.logits.argmax(dim=-1)
            if deterministic
            else distribution.sample()
        )
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            value,
        )

    def evaluate_actions(
        self,
        observation: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution(observation)
        return distribution.log_prob(actions), distribution.entropy(), value


def observation_to_torch(
    observation: Dict[str, np.ndarray],
    device: torch.device,
    *,
    add_batch: bool = True,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in observation.items():
        tensor = torch.as_tensor(value, device=device)
        if key != "action_mask":
            tensor = tensor.float()
        else:
            tensor = tensor.bool()
        if add_batch:
            tensor = tensor.unsqueeze(0)
        out[key] = tensor
    return out


def save_policy_checkpoint(
    path: Path,
    model: CandidatePolicyNet,
    *,
    kind: str,
    step: int,
    metrics: Dict[str, Any],
    optimizer: Any = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "kind": kind,
        "step": int(step),
        "model_config": model.config,
        "model": model.state_dict(),
        "metrics": metrics,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_policy_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> Tuple[CandidatePolicyNet, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CandidatePolicyNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model, checkpoint
