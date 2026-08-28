#!/usr/bin/env python3
"""Consistency baseline: how well does the FORMULA evaluator agree with humans?

Run this on labeled T1 (single-view grades) and/or T2 (pairwise preferences).
The learned quality dimension must beat these numbers before it ships.

T1 labels file (jsonl), one per line:
  {"sample_id": "...", "state": "path/to/state.json", "label": {"grade": 4}}
T2 labels file (jsonl), one per line — paths relative to the manifest dir:
  {"pair_id": "...", "state_a": "...", "state_b": "...", "label": {"prefer": "a|b|tie"}}
Both formats match the annotation tool export and pairs manifest.

Score used: the service 0-100 composite (score_adapt.build_score_card) when
--service-root is importable, else fallback score = -P.

Usage:
  python scripts/evalset_baseline_consistency.py --t2 data/evalsys/pairs_perturb/manifest.jsonl
  python scripts/evalset_baseline_consistency.py --t1 data/evalsys/t1_labels.jsonl --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.config import load_config
from layout_agent.evaluate import Evaluator
from layout_agent.models import LayoutState
from layout_agent.perceive import Perceiver

_SCORE_CARD = None


def _try_import_score_adapt(service_root: Path) -> None:
    global _SCORE_CARD
    src = service_root / "src"
    if (src / "score_adapt.py").is_file():
        sys.path.insert(0, str(src))
        try:
            from score_adapt import build_score_card  # noqa: PLC0415

            _SCORE_CARD = build_score_card
            print(f"using service score_adapt from {src}")
        except Exception as e:  # noqa: BLE001
            print(f"score_adapt import failed ({e}); falling back to -P")


def _score(report: Any) -> float:
    if _SCORE_CARD is not None:
        return float(_SCORE_CARD(report)["score"])
    return -float(report.metrics.get("P", 0.0))


class StateScorer:
    def __init__(self) -> None:
        cfg = load_config(ROOT / "config" / "default.yaml")
        self.perceiver = Perceiver(cfg)
        self.evaluator = Evaluator(cfg)
        self.work = ROOT / "data" / "evalsys" / "_consistency_work"
        self.work.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, float] = {}

    def score_file(self, state_path: Path) -> Optional[float]:
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
            else:
                sid = state.state_id or "v"
                state = self.perceiver.render_icm(
                    state, self.work / f"{sid}.instance.png", self.work / f"{sid}.obstacle.png"
                )
            report = self.evaluator.evaluate(state)
            val = _score(report)
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
            sp = Path(item["state"])
            if not sp.is_absolute():
                sp = path.parent / sp
            s = scorer.score_file(sp)
            if s is None:
                continue
            g = int(item["label"]["grade"])
            grades.append(float(g))
            scores.append(s)
            per_grade.setdefault(g, []).append(s)
    out = {
        "n": len(grades),
        "spearman": round(spearman(scores, grades), 4) if len(grades) >= 3 else None,
        "mean_score_by_grade": {
            str(g): round(sum(v) / len(v), 2) for g, v in sorted(per_grade.items())
        },
    }
    return out


def eval_t2(path: Path, scorer: StateScorer, margin: float = 1e-6) -> Dict[str, Any]:
    n = agree = disagree = model_tie = human_tie = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prefer = str(item.get("label", {}).get("prefer", "")).lower()
            if prefer not in ("a", "b", "tie"):
                continue
            base = path.parent
            sa = scorer.score_file(base / item["state_a"])
            sb = scorer.score_file(base / item["state_b"])
            if sa is None or sb is None:
                continue
            n += 1
            if prefer == "tie":
                human_tie += 1
                continue
            if abs(sa - sb) <= margin:
                model_tie += 1
                continue
            model_prefer = "a" if sa > sb else "b"
            if model_prefer == prefer:
                agree += 1
            else:
                disagree += 1
    decided = agree + disagree
    return {
        "n": n,
        "human_tie": human_tie,
        "model_tie": model_tie,
        "agree": agree,
        "disagree": disagree,
        "agreement_rate": round(agree / decided, 4) if decided else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", help="T1 labels jsonl (grades)")
    ap.add_argument("--t2", help="T2 labels jsonl (preferences) or pairs manifest.jsonl")
    ap.add_argument("--service-root", default=str(ROOT.parent / "service_ai_layout_agent"))
    ap.add_argument("--out", help="write report json here")
    args = ap.parse_args()
    if not args.t1 and not args.t2:
        ap.error("need --t1 and/or --t2")

    _try_import_score_adapt(Path(args.service_root))
    scorer = StateScorer()
    report: Dict[str, Any] = {"score_source": "score_adapt" if _SCORE_CARD else "-P"}
    if args.t1:
        report["t1"] = eval_t1(Path(args.t1), scorer)
    if args.t2:
        report["t2"] = eval_t2(Path(args.t2), scorer)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
