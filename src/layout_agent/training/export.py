"""Export a reward-model checkpoint to ONNX and verify onnxruntime parity.

Usage:
  python -m layout_agent.training.export --ckpt runs/reward_v0/best.pt --out runs/reward_v0/model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from layout_agent.training.checkpoint import strip_ddp_prefix
from layout_agent.training.model import RewardModel

# the dynamo exporter prints emoji; GBK Windows consoles crash without this
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def export_onnx(ckpt_path: Path, out_path: Path, opset: int = 18) -> Path:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = RewardModel.from_cfg(cfg, pretrained=False, freeze_stages=0)
    model.load_state_dict(strip_ddp_prefix(ckpt["model"]))
    model.eval()
    if int(cfg.get("feat_dim", 0)) > 0:
        raise SystemExit("formula-residual checkpoints need formula features; skip ONNX export")

    dummy = torch.zeros(1, cfg["channels"], cfg["img_size"], cfg["img_size"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["icm"],
        output_names=["logit"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        opset_version=opset,
    )

    # parity check
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = np.random.rand(2, cfg["channels"], cfg["img_size"], cfg["img_size"]).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"icm": x})[0].reshape(-1)
    diff = float(np.abs(ref - got).max())
    print(f"onnx parity max diff: {diff:.2e}")
    if diff > 1e-3:
        raise RuntimeError(f"onnx parity check failed: {diff}")
    return out_path


class OnnxScorer:
    """Lightweight inference wrapper for service integration (no torch needed)."""

    def __init__(self, onnx_path: Path, img_size: int, channels: int = 3):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.img_size = img_size
        self.channels = channels

    def score(self, arr: np.ndarray) -> float:
        """arr: uint8 [C,H,W] from training.data.encode_channels -> 0-100."""
        x = (arr.astype(np.float32) / 255.0)[None]
        logit = float(self.sess.run(None, {"icm": x})[0].reshape(-1)[0])
        return 100.0 / (1.0 + float(np.exp(-logit)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = export_onnx(Path(args.ckpt), Path(args.out))
    print(f"exported: {p}")


if __name__ == "__main__":
    main()
