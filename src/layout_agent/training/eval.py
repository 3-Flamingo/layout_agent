"""Evaluate a reward-model checkpoint on labeled T1/T2 (same protocol as the
formula baseline script, so numbers are directly comparable).

Usage:
  python -m layout_agent.training.eval --ckpt runs/reward_v0/best.pt \
      --t2 data/evalsys/pairs_perturb/manifest.jsonl [--t1 t1_labels.jsonl] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from layout_agent.models import LayoutState
from layout_agent.training.checkpoint import strip_ddp_prefix
from layout_agent.training.data import StateEncoder, encode_channels, _obstacle_for
from layout_agent.training.metrics import pairwise_agreement, spearman
from layout_agent.training.model import RewardModel


class CheckpointScorer:
    def __init__(self, ckpt_path: Path, device: Optional[str] = None):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = ckpt["cfg"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = RewardModel(self.cfg["channels"], 0, False).to(self.device)
        self.model.load_state_dict(strip_ddp_prefix(ckpt["model"]))
        self.model.eval()
        self.encoder = StateEncoder(self.cfg["img_size"], self.cfg["channels"])
        self._cache: Dict[str, float] = {}

    @torch.no_grad()
    def _score_arr(self, arr: np.ndarray) -> float:
        x = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
        logit = self.model(x)
        return float(RewardModel.to_score(logit).item())

    def score_state_file(self, path: Path) -> Optional[float]:
        key = str(Path(path).resolve())
        if key in self._cache:
            return self._cache[key]
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            state = LayoutState.model_validate(raw)
            arr = self.encoder.encode(state, base_dir=Path(path).parent.parent)
            val = self._score_arr(arr)
        except Exception as e:  # noqa: BLE001
            print(f"score failed {Path(path).name}: {e}")
            return None
        self._cache[key] = val
        return val

    def score_icm(self, instance_png: Path, state: Optional[LayoutState]) -> float:
        arr = encode_channels(
            instance_png, _obstacle_for(instance_png), state,
            self.cfg["img_size"], self.cfg["channels"],
        )
        return self._score_arr(arr)


def _resolve(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else base / q


def eval_t2(path: Path, scorer: CheckpointScorer) -> Dict:
    scored: List[Tuple[float, float, str]] = []
    by_level: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prefer = str(rec.get("label", {}).get("prefer", "")).lower()
        if prefer not in ("a", "b", "tie"):
            continue
        sa = scorer.score_state_file(_resolve(path.parent, rec["state_a"]))
        sb = scorer.score_state_file(_resolve(path.parent, rec["state_b"]))
        if sa is None or sb is None:
            continue
        scored.append((sa, sb, prefer))
        if rec.get("level"):
            by_level[rec["level"]].append((sa, sb, prefer))
    out = {"overall": pairwise_agreement(scored)}
    for lv, s in sorted(by_level.items()):
        out[lv] = pairwise_agreement(s)
    return out


def eval_t1(path: Path, scorer: CheckpointScorer) -> Dict:
    grades: List[float] = []
    scores: List[float] = []
    per_grade: Dict[int, List[float]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        s = scorer.score_state_file(_resolve(path.parent, rec["state"]))
        if s is None:
            continue
        g = int(rec["label"]["grade"])
        grades.append(float(g))
        scores.append(s)
        per_grade[g].append(s)
    return {
        "n": len(grades),
        "spearman": round(spearman(scores, grades), 4),
        "mean_score_by_grade": {str(g): round(sum(v) / len(v), 2) for g, v in sorted(per_grade.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--t1")
    ap.add_argument("--t2")
    ap.add_argument("--out")
    ap.add_argument("--device")
    args = ap.parse_args()
    if not args.t1 and not args.t2:
        ap.error("need --t1 and/or --t2")

    scorer = CheckpointScorer(Path(args.ckpt), args.device)
    report: Dict = {"ckpt": args.ckpt}
    if args.t2:
        report["t2"] = eval_t2(Path(args.t2), scorer)
    if args.t1:
        report["t1"] = eval_t1(Path(args.t1), scorer)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
