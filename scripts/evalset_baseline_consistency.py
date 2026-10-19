#!/usr/bin/env python3
"""Consistency baseline: how well does the FORMULA evaluator agree with humans?

Run this on labeled T1 (single-view grades) and/or T2 (pairwise preferences).
The learned quality dimension must beat these numbers before it ships.

T1 labels file (jsonl), one per line:
  {"sample_id": "...", "state": "path/to/state.json", "label": {"grade": 4}}
T2 labels file (jsonl), one per line — paths relative to the manifest dir:
  {"pair_id": "...", "state_a": "...", "state_b": "...", "label": {"prefer": "a|b|tie"}}
Both formats match the annotation tool export and pairs manifest.

Score used: unrounded ``score_adapt.ranking_key`` for forced-choice T2 (new
human_pairs protocol has no 「差不多」). Display 0-100
``build_score_card`` is kept for T1 and for ``--no-force-choice``. Fallback: -P.

Usage:
  python scripts/evalset_baseline_consistency.py --t2 data/evalsys/pairs_perturb/manifest.jsonl
  python scripts/evalset_baseline_consistency.py --t1 data/evalsys/t1_labels.jsonl --out report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.config import load_config
from layout_agent.evaluate import Evaluator
from layout_agent.models import LayoutState
from layout_agent.perceive import Perceiver
from layout_agent.training.metrics import pairwise_agreement

_SCORE_CARD = None
_RANK_KEY = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _try_import_score_adapt(service_root: Path) -> None:
    global _SCORE_CARD, _RANK_KEY
    src = service_root / "src"
    if (src / "score_adapt.py").is_file():
        sys.path.insert(0, str(src))
        try:
            from score_adapt import build_score_card, ranking_key  # noqa: PLC0415

            _SCORE_CARD = build_score_card
            _RANK_KEY = ranking_key
            print(f"using service score_adapt from {src}")
        except Exception as e:  # noqa: BLE001
            print(f"score_adapt import failed ({e}); falling back to -P")


@dataclass(frozen=True)
class ScoredState:
    display: float
    rank: Tuple[float, ...]


def _ann_spread(state: LayoutState) -> float:
    """Mean distance of annotation poses from their centroid (smaller is tighter)."""
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
    return sum(((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 for p in pts) / len(pts)


def _score_report(report: Any, state: Optional[LayoutState] = None) -> ScoredState:
    p = -float(report.metrics.get("P", 0.0))
    if _SCORE_CARD is None and _RANK_KEY is None:
        extra = (-_ann_spread(state),) if state is not None else ()
        return ScoredState(display=p, rank=(p,) + extra)
    display = float(_SCORE_CARD(report)["score"]) if _SCORE_CARD is not None else p
    rank = tuple(_RANK_KEY(report)) if _RANK_KEY is not None else (display,)
    if state is not None:
        rank = rank + (-_ann_spread(state),)
    return ScoredState(display=display, rank=rank)


def _resolve_state_path(rel: str, jsonl_path: Path) -> Path:
    """Resolve state paths from either the jsonl directory or the repo root."""
    p = Path(rel)
    if p.is_absolute():
        return p
    for cand in (jsonl_path.parent / p, ROOT / p, p):
        if cand.exists():
            return cand
    return jsonl_path.parent / p


class StateScorer:
    def __init__(self) -> None:
        cfg = load_config(ROOT / "config" / "default.yaml")
        self.perceiver = Perceiver(cfg)
        self.evaluator = Evaluator(cfg)
        self.work = ROOT / "data" / "evalsys" / "_consistency_work"
        self.work.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, ScoredState] = {}

    def score_file(self, state_path: Path) -> Optional[ScoredState]:
        key = str(state_path.resolve())
        if key in self._cache:
            return self._cache[key]
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            state = LayoutState.model_validate(raw)
            inst = raw.get("raster", {}).get("instance_image") or ""
            inst_p = Path(inst)
            if not inst_p.is_absolute():
                inst_p = state_path.parent.parent / inst  # states/../ = pairs root
            if inst_p.exists():
                state.raster.instance_image = str(inst_p.resolve())
                obs = raw.get("raster", {}).get("obstacle_image") or ""
                obs_p = Path(obs) if Path(obs).is_absolute() else state_path.parent.parent / obs
                if obs and obs_p.exists():
                    state.raster.obstacle_image = str(obs_p.resolve())
            # Geometry evaluate does not need ICM rasters; skip slow re-render.
            report = self.evaluator.evaluate(state)
            val = _score_report(report, state)
        except Exception as e:  # noqa: BLE001
            print(f"score failed for {state_path.name}: {e}")
            val = None
        if val is not None:
            self._cache[key] = val
        return val


def spearman(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation with average ranks for ties (no scipy)."""
    def ranks(v: List[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx * dy == 0:
        return 0.0
    return num / (dx * dy)


def eval_t1(path: Path, scorer: StateScorer) -> Dict[str, Any]:
    grades: List[float] = []
    scores: List[float] = []
    per_grade: Dict[int, List[float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sp = _resolve_state_path(str(item["state"]), path)
            s = scorer.score_file(sp)
            if s is None:
                continue
            g = int(item["label"]["grade"])
            grades.append(float(g))
            scores.append(s.display)
            per_grade.setdefault(g, []).append(s.display)
    out = {
        "n": len(grades),
        "spearman": round(spearman(scores, grades), 4) if len(grades) >= 3 else None,
        "mean_score_by_grade": {
            str(g): round(sum(v) / len(v), 2) for g, v in sorted(per_grade.items())
        },
    }
    return out


def eval_t2(
    path: Path,
    scorer: StateScorer,
    margin: float = 1e-6,
    split: str = "",
    *,
    force_choice: bool = True,
) -> Dict[str, Any]:
    scored: List[Tuple[Any, Any, str]] = []
    by_level: Dict[str, List[Tuple[Any, Any, str]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if split and item.get("split") != split:
                continue
            prefer = str(item.get("label", {}).get("prefer", "")).lower()
            if prefer not in ("a", "b", "tie"):
                continue
            sa = scorer.score_file(_resolve_state_path(str(item["state_a"]), path))
            sb = scorer.score_file(_resolve_state_path(str(item["state_b"]), path))
            if sa is None or sb is None:
                continue
            if force_choice:
                row = (sa.rank, sb.rank, prefer)
            else:
                row = (sa.display, sb.display, prefer)
            scored.append(row)
            level = str(item.get("level") or "")
            if level:
                by_level[level].append(row)
    out: Dict[str, Any] = {
        "overall": pairwise_agreement(scored, margin, force_choice=force_choice),
        "decision_protocol": "forced_choice" if force_choice else "margin_tie",
    }
    for level, rows in sorted(by_level.items()):
        out[level] = pairwise_agreement(rows, margin, force_choice=force_choice)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", help="T1 labels jsonl (grades)")
    ap.add_argument("--t2", help="T2 labels jsonl (preferences) or pairs manifest.jsonl")
    ap.add_argument("--split", default="", help="T2 split to evaluate, e.g. val or challenge")
    ap.add_argument("--margin", type=float, default=1e-6, help="absolute score difference treated as a tie")
    ap.add_argument(
        "--force-choice",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pick a side whenever scores differ (new human_pairs protocol; no 「差不多」)",
    )
    ap.add_argument("--service-root", default=str(ROOT.parent / "service_ai_layout_agent"))
    ap.add_argument("--out", help="write report json here")
    args = ap.parse_args()
    if not args.t1 and not args.t2:
        ap.error("need --t1 and/or --t2")

    _try_import_score_adapt(Path(args.service_root))
    scorer = StateScorer()
    if args.force_choice and _RANK_KEY is not None:
        score_source = "score_adapt.ranking_key"
    elif _SCORE_CARD is not None:
        score_source = "score_adapt"
    else:
        score_source = "-P"
    report: Dict[str, Any] = {
        "score_source": score_source,
        "decision_protocol": "forced_choice" if args.force_choice else "margin_tie",
        "inputs": {},
    }
    if args.split:
        report["split"] = args.split
    if args.t1:
        t1_path = Path(args.t1)
        report["inputs"]["t1"] = {
            "path": str(t1_path.resolve()),
            "sha256": _sha256(t1_path),
        }
        report["t1"] = eval_t1(t1_path, scorer)
    if args.t2:
        t2_path = Path(args.t2)
        report["inputs"]["t2"] = {
            "path": str(t2_path.resolve()),
            "sha256": _sha256(t2_path),
        }
        report["t2"] = eval_t2(
            t2_path,
            scorer,
            args.margin,
            args.split,
            force_choice=args.force_choice,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
