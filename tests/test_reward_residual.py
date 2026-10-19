from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from layout_agent.models import EvalReport
from layout_agent.training.data import encode_channels
from layout_agent.training.formula_features import FORMULA_FEAT_DIM, features_from_report
from layout_agent.training.model import RewardModel


class FormulaFeatureTest(unittest.TestCase):
    def test_feature_dim_matches_constant(self) -> None:
        report = EvalReport(
            state_id="x",
            hard_fail=True,
            metrics={
                "P": 12.0,
                "V_hard": 2.0,
                "S_layer": 1.0,
                "PathLen": 40.0,
                "Overcrowd": 0.0,
                "S_balance": 1.0,
            },
            side_density={"L": 2.0, "R": 0.0, "U": 1.0, "D": 1.0},
        )
        feat, score, display = features_from_report(report)
        self.assertEqual(feat.shape, (FORMULA_FEAT_DIM,))
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertEqual(display, round(score, 2))


class ResidualModelTest(unittest.TestCase):
    def test_zero_init_residual_equals_formula_score(self) -> None:
        model = RewardModel(in_channels=5, freeze_stages=0, pretrained=False, feat_dim=FORMULA_FEAT_DIM)
        model.eval()
        x = torch.zeros(3, 5, 64, 64)
        feats = torch.zeros(3, FORMULA_FEAT_DIM)
        formula = torch.tensor([12.0, 55.5, 90.0])
        with torch.no_grad():
            out = model(x, feats, formula)
        self.assertTrue(torch.allclose(out, formula, atol=1e-5))


class EncodeChannelsTest(unittest.TestCase):
    def test_five_channel_keeps_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inst = Path(tmp) / "x.instance.png"
            rgb = np.zeros((32, 32, 3), dtype=np.uint8)
            rgb[8:24, 8:24] = (10, 200, 30)
            Image.fromarray(rgb).save(inst)
            arr = encode_channels(inst, None, None, img_size=16, channels=5)
            self.assertEqual(arr.shape, (5, 16, 16))
            self.assertGreater(int(arr[1].max()), 0)


if __name__ == "__main__":
    unittest.main()
