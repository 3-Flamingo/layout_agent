"""Evaluate a reward-model checkpoint on labeled T1/T2 (same protocol as the
formula baseline script, so numbers are directly comparable).

Usage:
  python -m layout_agent.training.eval --ckpt runs/reward_v0/best.pt \
      --t2 data/evalsys/pairs_perturb/manifest.jsonl [--t1 t1_labels.jsonl] [--out report.json]
"""

from __future__ import annotations

import argparse
import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointScorer:
    def __init__(self, ckpt_path: Path, device: Optional[str] = None):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = ckpt["cfg"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.formula_residual = bool(self.cfg.get("formula_residual")) and int(self.cfg.get("feat_dim", 0)) > 0
        self.model = RewardModel.from_cfg(self.cfg, pretrained=False, freeze_stages=0).to(self.device)
        self.model.load_state_dict(strip_ddp_prefix(ckpt["model"]))
        self.model.eval()
        self.encoder = StateEncoder(self.cfg["img_size"], self.cfg["channels"])
        self._cache: Dict[str, float] = {}
        self._logit_cache: Dict[str, float] = {}
        self._extractor = None
        if self.formula_residual:
            from layout_agent.training.formula_features import FormulaExtractor

            self._extractor = FormulaExtractor()

    def _pack_state(self, state: LayoutState) -> Tuple[Optional[np.ndarray], Optional[float]]:
        if self._extractor is None:
            return None, None
        packed = self._extractor.pack_state(state)
        return np.asarray(packed["feat"], dtype=np.float32), float(packed["formula_score"])

    @torch.no_grad()
    def _forward_arr(
        self,
        arr: np.ndarray,
        feat: Optional[np.ndarray] = None,
        formula_score: Optional[float] = None,
    ) -> float:
        x = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
        feats_t = None
        fs_t = None
        if self.formula_residual:
            if feat is None or formula_score is None:
                raise RuntimeError("formula-residual scorer needs features")
            feats_t = torch.from_numpy(np.asarray(feat, dtype=np.float32)).unsqueeze(0).to(self.device)
            fs_t = torch.tensor([float(formula_score)], dtype=torch.float32, device=self.device)
        amp_name = str(self.cfg.get("amp", "off")).lower()
        amp_dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
        use_amp = self.device.type == "cuda" and amp_name in ("bf16", "fp16")
        with torch.amp.autocast(self.device.type, enabled=use_amp, dtype=amp_dtype):
            out = self.model(x, feats_t, fs_t)
        return float(out.item())

    def _logit_arr(self, arr: np.ndarray) -> float:
        return self._forward_arr(arr)

    def _score_arr(self, arr: np.ndarray) -> float:
        logit = self._logit_arr(arr)
        return float(RewardModel.to_score(torch.tensor(logit)).item())

    def score_parts(
        self,
        arr: np.ndarray,
        state: Optional[LayoutState] = None,
    ) -> Dict[str, float]:
        """Single-layout score. Combined ranking score is `score` (unclipped)."""
        feat, formula = (None, None)
        if self.formula_residual:
            if state is None:
                raise RuntimeError("formula-residual scoring needs a LayoutState")
            feat, formula = self._pack_state(state)
            combined = self._forward_arr(arr, feat, formula)
            residual = combined - float(formula)
            display = float(max(0.0, min(100.0, combined)))
            return {
                "score": float(combined),
                "display": display,
                "formula_score": float(formula),
                "residual": float(residual),
            }
        logit = self._forward_arr(arr)
        display = float(RewardModel.to_score(torch.tensor(logit)).item())
        return {"score": logit, "display": display, "formula_score": display, "residual": 0.0}

    def score_state(self, state: LayoutState) -> float:
        """0-100 display score for a state whose ICM is already rendered."""
        parts = self.score_parts(self.encoder.encode(state), state)
        return float(parts["display"])

    def score_state_file(self, path: Path) -> Optional[float]:
        key = str(Path(path).resolve())
        if key in self._cache:
            return self._cache[key]
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            state = LayoutState.model_validate(raw)
            arr = self.encoder.encode(state, base_dir=Path(path).parent.parent)
            val = float(self.score_parts(arr, state)["display"])
        except Exception as e:  # noqa: BLE001
            print(f"score failed {Path(path).name}: {e}")
            return None
        self._cache[key] = val
        return val

    def logit_state_file(self, path: Path) -> Optional[float]:
        key = str(Path(path).resolve())
        if key in self._logit_cache:
            return self._logit_cache[key]
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            state = LayoutState.model_validate(raw)
            arr = self.encoder.encode(state, base_dir=Path(path).parent.parent)
            val = float(self.score_parts(arr, state)["score"])
        except Exception as e:  # noqa: BLE001
            print(f"score failed {Path(path).name}: {e}")
            return None
        self._logit_cache[key] = val
        return val

    def score_icm(self, instance_png: Path, state: Optional[LayoutState]) -> float:
        arr = encode_channels(
            instance_png, _obstacle_for(instance_png), state,
            self.cfg["img_size"], self.cfg["channels"],
        )
        return float(self.score_parts(arr, state)["display"])

    def logit_pair_side(
        self, rec: Dict, which: str, manifest_dir: Path
    ) -> Optional[float]:
        """Ranking score for the exact pre-rendered representation used for training."""
        state_path = _resolve(manifest_dir, rec[f"state_{which}"])
        icm_raw = rec.get(f"icm_{which}")
        icm_path = _resolve(manifest_dir, icm_raw) if icm_raw else None
        key = f"rank:{icm_path.resolve() if icm_path else state_path.resolve()}|{state_path.resolve()}|c{self.cfg.get('channels')}"
        if key in self._logit_cache:
            return self._logit_cache[key]
        try:
            state = None
            if state_path.exists():
                raw = json.loads(state_path.read_text(encoding="utf-8"))
                state = LayoutState.model_validate(raw)
            if icm_path is not None and icm_path.exists():
                arr = encode_channels(
                    icm_path,
                    _obstacle_for(icm_path),
                    state,
                    self.cfg["img_size"],
                    self.cfg["channels"],
                )
            else:
                arr = self.encoder.encode(state, base_dir=state_path.parent.parent) if state else None
                if arr is None:
                    return None
            val = float(self.score_parts(arr, state)["score"])
        except Exception as e:  # noqa: BLE001
            print(f"score failed pair {rec.get('pair_id')} {which}: {e}")
            return None
        self._logit_cache[key] = val
        return val


def _resolve(base: Path, p: str) -> Path:
    from layout_agent.training.data import resolve_pair_path

    return resolve_pair_path(base, p)


def eval_t2(
    path: Path,
    scorer: CheckpointScorer,
    split: str = "",
    margin: float = 1e-6,
) -> Dict:
    scored: List[Tuple[float, float, str]] = []
    by_level: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if split and rec.get("split") != split:
            continue
        prefer = str(rec.get("label", {}).get("prefer", "")).lower()
        if prefer not in ("a", "b", "tie"):
            continue
        sa = scorer.logit_pair_side(rec, "a", path.parent)
        sb = scorer.logit_pair_side(rec, "b", path.parent)
        if sa is None or sb is None:
            continue
        scored.append((sa, sb, prefer))
        if rec.get("level"):
            by_level[rec["level"]].append((sa, sb, prefer))
    out = {"overall": pairwise_agreement(scored, margin, force_choice=True)}
    for lv, s in sorted(by_level.items()):
        out[lv] = pairwise_agreement(s, margin, force_choice=True)
    return out


def eval_t1(path: Path, scorer: CheckpointScorer) -> Dict:
    grades: List[float] = []
    scores: List[float] = []
    per_grade: Dict[int, List[float]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        s = scorer.logit_state_file(_resolve(path.parent, rec["state"]))
        if s is None:
            continue
        g = int(rec["label"]["grade"])
        grades.append(float(g))
        scores.append(s)
        if scorer.formula_residual:
            display = float(max(0.0, min(100.0, s)))
        else:
            display = float(RewardModel.to_score(torch.tensor(s)).item())
        per_grade[g].append(display)
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
    ap.add_argument("--split", default="", help="T2 split to evaluate, e.g. val or challenge")
    ap.add_argument("--margin", type=float, default=1e-6, help="absolute score difference treated as a tie")
    ap.add_argument("--out")
    ap.add_argument("--device")
    args = ap.parse_args()
    if not args.t1 and not args.t2:
        ap.error("need --t1 and/or --t2")

    scorer = CheckpointScorer(Path(args.ckpt), args.device)
    checkpoint_path = Path(args.ckpt)
    report: Dict = {
        "ckpt": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "formula_residual": bool(scorer.formula_residual),
        "decision_protocol": "forced_choice",
        "inputs": {},
    }
    if args.split:
        report["split"] = args.split
    if args.t2:
        t2_path = Path(args.t2)
        report["inputs"]["t2"] = {
            "path": str(t2_path.resolve()),
            "sha256": _sha256(t2_path),
        }
        report["t2"] = eval_t2(t2_path, scorer, args.split, args.margin)
    if args.t1:
        t1_path = Path(args.t1)
        report["inputs"]["t1"] = {
            "path": str(t1_path.resolve()),
            "sha256": _sha256(t1_path),
        }
        report["t1"] = eval_t1(t1_path, scorer)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
