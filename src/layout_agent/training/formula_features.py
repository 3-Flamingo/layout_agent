"""Per-state formula features and scalar quality for residual reward training.

The ranking scalar is the unrounded overall 0–100 used by score_adapt.ranking_key.
Extra vector components let the residual head see hard counts, path length, and
side density without re-learning constraints from pixels.

Keep the overall blend in sync with service_ai_layout_agent/src/score_adapt.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from layout_agent.config import load_config
from layout_agent.evaluate import Evaluator
from layout_agent.models import LayoutState

_REPO_ROOT = Path(__file__).resolve().parents[3]

_DEFAULT_SCALES: Dict[str, float] = {
    "hard": 1.0,
    "layer_order": 2.0,
    "path_len": 80.0,
    "overcrowd": 2.0,
    "side_balance": 2.0,
}

_OVERALL_WEIGHTS: Dict[str, float] = {
    "hard": 0.50,
    "layer_order": 0.18,
    "path_len": 0.12,
    "overcrowd": 0.10,
    "side_balance": 0.10,
}

FORMULA_FEAT_NAMES: Tuple[str, ...] = (
    "overall_01",
    "log1p_P",
    "log1p_V_hard",
    "S_layer",
    "path_len_n",
    "overcrowd",
    "S_balance",
    "hard_fail",
    "dens_L",
    "dens_R",
    "dens_U",
    "dens_D",
    "dens_var",
    "n_empty_n",
    "max_side_n",
    "ann_spread_n",
)
FORMULA_FEAT_DIM = len(FORMULA_FEAT_NAMES)

CACHE_VERSION = 1


def _cost_to_score(raw: float, scale: float) -> float:
    r = max(0.0, float(raw))
    s = max(1e-9, float(scale))
    return 100.0 / (1.0 + r / s)


def ranking_key(report: Any) -> Tuple[float, ...]:
    """Higher is better. Same blend as score_adapt.ranking_key (unrounded)."""
    metrics = dict(getattr(report, "metrics", None) or {})
    violations = list(getattr(report, "hard_violations", None) or [])
    v_hard = float(metrics.get("V_hard", len(violations)))
    s_layer = float(metrics.get("S_layer", 0.0))
    path_len = float(metrics.get("PathLen", metrics.get("S_path", 0.0)))
    overcrowd = float(metrics.get("Overcrowd", metrics.get("S_overcrowd", 0.0)))
    balance = float(metrics.get("S_balance", 0.0))
    p_cost = float(metrics.get("P", 0.0))
    overall = (
        _OVERALL_WEIGHTS["hard"] * _cost_to_score(v_hard, _DEFAULT_SCALES["hard"])
        + _OVERALL_WEIGHTS["layer_order"] * _cost_to_score(s_layer, _DEFAULT_SCALES["layer_order"])
        + _OVERALL_WEIGHTS["path_len"] * _cost_to_score(path_len, _DEFAULT_SCALES["path_len"])
        + _OVERALL_WEIGHTS["overcrowd"] * _cost_to_score(overcrowd, _DEFAULT_SCALES["overcrowd"])
        + _OVERALL_WEIGHTS["side_balance"] * _cost_to_score(balance, _DEFAULT_SCALES["side_balance"])
    )
    if bool(getattr(report, "hard_fail", False)):
        overall = min(overall, _cost_to_score(v_hard, _DEFAULT_SCALES["hard"]))
    dens = dict(getattr(report, "side_density", None) or {})
    vals = [float(dens.get(k, 0.0)) for k in ("L", "R", "U", "D")]
    mean = sum(vals) / 4.0
    var = sum((x - mean) ** 2 for x in vals) / 4.0
    n_empty = float(sum(1 for x in vals if x <= 0.0))
    max_side = max(vals) if vals else 0.0
    return (
        float(overall),
        -p_cost,
        -v_hard,
        -s_layer,
        -path_len,
        -overcrowd,
        -balance,
        -var,
        -n_empty,
        -max_side,
    )


def ann_spread(state: Optional[LayoutState]) -> float:
    if state is None:
        return 0.0
    pts: List[Tuple[float, float]] = []
    for lin in state.linears:
        pt = lin.text_point or lin.point
        pts.append((float(pt.x), float(pt.y)))
    for rad in list(state.diameters) + list(state.radials):
        if rad.path:
            pts.append((float(rad.path[-1].x), float(rad.path[-1].y)))
        else:
            pts.append((float(rad.anchor.x), float(rad.anchor.y)))
    if len(pts) < 2:
        return 0.0
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return sum(math.hypot(p[0] - cx, p[1] - cy) for p in pts) / len(pts)


def features_from_report(
    report: Any, state: Optional[LayoutState] = None
) -> Tuple[np.ndarray, float, float]:
    """Return (feat[FORMULA_FEAT_DIM], formula_score 0-100, display rounded)."""
    metrics = dict(getattr(report, "metrics", None) or {})
    key = ranking_key(report)
    overall = float(key[0])
    v_hard = float(metrics.get("V_hard", 0.0))
    p_cost = float(metrics.get("P", 0.0))
    s_layer = float(metrics.get("S_layer", 0.0))
    path_len = float(metrics.get("PathLen", metrics.get("S_path", 0.0)))
    overcrowd = float(metrics.get("Overcrowd", metrics.get("S_overcrowd", 0.0)))
    balance = float(metrics.get("S_balance", 0.0))
    dens = dict(getattr(report, "side_density", None) or {})
    vals = [float(dens.get(k, 0.0)) for k in ("L", "R", "U", "D")]
    mean = sum(vals) / 4.0
    var = sum((x - mean) ** 2 for x in vals) / 4.0
    n_empty = float(sum(1 for x in vals if x <= 0.0))
    max_side = max(vals) if vals else 0.0
    spread = ann_spread(state)
    feat = np.array(
        [
            overall / 100.0,
            math.log1p(max(0.0, p_cost)),
            math.log1p(max(0.0, v_hard)),
            s_layer,
            path_len / _DEFAULT_SCALES["path_len"],
            overcrowd,
            balance,
            1.0 if bool(getattr(report, "hard_fail", False)) else 0.0,
            vals[0] / 10.0,
            vals[1] / 10.0,
            vals[2] / 10.0,
            vals[3] / 10.0,
            var,
            n_empty / 4.0,
            max_side / 10.0,
            spread / 50.0,
        ],
        dtype=np.float32,
    )
    if feat.shape != (FORMULA_FEAT_DIM,):
        raise RuntimeError(f"formula feat dim {feat.shape} != {FORMULA_FEAT_DIM}")
    return feat, overall, round(overall, 2)


def pair_side_key(pair_id: str, side: str) -> str:
    return f"{pair_id}|{side}"


class FormulaExtractor:
    """Geometry evaluator + feature packer (no ICM required)."""

    def __init__(self, cfg_path: Optional[Path] = None) -> None:
        cfg = load_config(cfg_path or (_REPO_ROOT / "config" / "default.yaml"))
        self.evaluator = Evaluator(cfg)

    def pack_state(self, state: LayoutState) -> Dict[str, Any]:
        report = self.evaluator.evaluate(state)
        feat, formula_score, display = features_from_report(report, state)
        return {
            "feat": feat.tolist(),
            "formula_score": float(formula_score),
            "display": float(display),
        }

    def pack_state_file(self, path: Path) -> Dict[str, Any]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        state = LayoutState.model_validate(raw)
        return self.pack_state(state)


class FormulaCache:
    """Portable pair_id|side -> features. Safe to share across Windows/Linux."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.data: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if int(raw.get("version", 0)) == CACHE_VERSION:
                self.data = dict(raw.get("by_pair_side") or {})

    def get(
        self,
        pair_id: str,
        side: str,
        state_path: Path,
        extractor: FormulaExtractor,
    ) -> Tuple[np.ndarray, float]:
        key = pair_side_key(pair_id, side)
        item = self.data.get(key)
        if item is None:
            item = extractor.pack_state_file(state_path)
            self.data[key] = item
        feat = np.asarray(item["feat"], dtype=np.float32)
        if feat.shape != (FORMULA_FEAT_DIM,):
            item = extractor.pack_state_file(state_path)
            self.data[key] = item
            feat = np.asarray(item["feat"], dtype=np.float32)
        return feat, float(item["formula_score"])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CACHE_VERSION, "by_pair_side": self.data}
        self.path.write_text(json.dumps(payload), encoding="utf-8")


def cache_path_for_manifest(manifest: Path) -> Path:
    return Path(manifest).resolve().parent / "formula_cache.json"


def precompute_manifest_formula(manifest: Path, *, extractor: Optional[FormulaExtractor] = None) -> Path:
    """Fill formula_cache.json for every pair side in the manifest."""
    from layout_agent.training.data import resolve_pair_path

    manifest = Path(manifest)
    cache = FormulaCache(cache_path_for_manifest(manifest))
    ext = extractor or FormulaExtractor()
    root = manifest.parent
    n_need = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        pid = str(rec.get("pair_id") or "")
        if not pid:
            continue
        for side in ("a", "b"):
            sk = rec.get(f"state_{side}")
            if not sk:
                continue
            key = pair_side_key(pid, side)
            if key in cache.data:
                continue
            cache.get(pid, side, resolve_pair_path(root, str(sk)), ext)
            n_need += 1
            if n_need % 50 == 0:
                print(f"formula cache computed {n_need} new sides")
    cache.save()
    print(f"formula cache {cache.path} entries={len(cache.data)} new={n_need}")
    return cache.path
