"""Reward model: ResNet18 backbone -> scalar quality logit."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class RewardModel(nn.Module):
    """Outputs an unbounded logit; display score is sigmoid(logit) * 100.

    freeze_stages: 0 = train everything; 1 freezes conv1+bn1; 2 also layer1;
    3 also layer2 ... (useful while the dataset is only a few hundred pairs).
    """

    def __init__(self, in_channels: int = 3, freeze_stages: int = 0, pretrained: bool = False):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        net = resnet18(weights=weights)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.fc = nn.Linear(net.fc.in_features, 1)
        self.net = net
        self._freeze(freeze_stages)

    def _freeze(self, stages: int) -> None:
        groups = [
            [self.net.conv1, self.net.bn1],
            [self.net.layer1],
            [self.net.layer2],
            [self.net.layer3],
        ]
        for g in groups[:stages]:
            for m in g:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [B]

    @staticmethod
    def to_score(logit: torch.Tensor) -> torch.Tensor:
        """Map logit -> user-facing 0-100 quality score."""
        return torch.sigmoid(logit) * 100.0
