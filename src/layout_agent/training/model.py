"""Reward model: ResNet18 backbone -> scalar layout quality.

Two modes:
- image-only (feat_dim=0): unbounded logit; display = sigmoid(logit)*100.
- formula residual (feat_dim>0): score = formula_score + residual(img, feats).
  residual_head is zero-initialized so a new model ranks like the formula.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18


def _adapt_conv1(old: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
    if old.in_channels == in_channels:
        return old
    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        if pretrained and old.weight.shape[1] >= 1:
            n_copy = min(in_channels, old.weight.shape[1])
            new.weight[:, :n_copy] = old.weight[:, :n_copy]
            if in_channels > n_copy:
                fill = old.weight.mean(dim=1, keepdim=True)
                new.weight[:, n_copy:] = fill.repeat(1, in_channels - n_copy, 1, 1)
        else:
            nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
        if new.bias is not None:
            nn.init.zeros_(new.bias)
    return new


class RewardModel(nn.Module):
    """Outputs a ranking score (higher = better layout).

    freeze_stages: 0 = train everything; 1 freezes conv1+bn1; 2 also layer1;
    3 also layer2 ... (useful while the dataset is only a few hundred pairs).
    """

    def __init__(
        self,
        in_channels: int = 3,
        freeze_stages: int = 0,
        pretrained: bool = False,
        feat_dim: int = 0,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        weights = "IMAGENET1K_V1" if pretrained else None
        net = resnet18(weights=weights)
        if in_channels != 3:
            net.conv1 = _adapt_conv1(net.conv1, in_channels, pretrained)
        feat_ch = int(net.fc.in_features)
        net.fc = nn.Identity()
        self.backbone = net
        if self.feat_dim > 0:
            self.feat_mlp = nn.Sequential(
                nn.Linear(self.feat_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
            )
            self.residual_head = nn.Linear(feat_ch + 32, 1)
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)
            self.img_head = None
        else:
            self.feat_mlp = None
            self.residual_head = None
            self.img_head = nn.Linear(feat_ch, 1)
        self._freeze(freeze_stages)

    def _freeze(self, stages: int) -> None:
        groups = [
            [self.backbone.conv1, self.backbone.bn1],
            [self.backbone.layer1],
            [self.backbone.layer2],
            [self.backbone.layer3],
        ]
        for g in groups[:stages]:
            for m in g:
                for p in m.parameters():
                    p.requires_grad = False

    def image_feat(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def residual(self, x: torch.Tensor, feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.image_feat(x)
        if self.residual_head is None or feats is None:
            if self.img_head is None:
                raise RuntimeError("residual model requires formula features")
            return self.img_head(h).squeeze(-1)
        f = self.feat_mlp(feats)
        return self.residual_head(torch.cat([h, f], dim=-1)).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        formula_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r = self.residual(x, feats)
        if formula_score is not None:
            return formula_score.reshape(-1) + r
        return r

    @staticmethod
    def to_score(logit: torch.Tensor) -> torch.Tensor:
        """Map image-only logit -> user-facing 0-100 quality score."""
        return torch.sigmoid(logit) * 100.0

    @staticmethod
    def from_cfg(cfg: dict, *, pretrained: Optional[bool] = None, freeze_stages: Optional[int] = None) -> "RewardModel":
        pre = bool(cfg.get("pretrained", False) if pretrained is None else pretrained)
        freeze = int(cfg.get("freeze_stages", 0) if freeze_stages is None else freeze_stages)
        return RewardModel(
            int(cfg.get("channels", 3)),
            freeze,
            pre,
            feat_dim=int(cfg.get("feat_dim", 0)),
        )
