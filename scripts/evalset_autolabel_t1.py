#!/usr/bin/env python3
"""T1 auto labels for the annotation pack.

  - colleague perfect  → 5
  - colleague perturbed → 2
  - production snapshots → grade from visible hard constraints
    (text overlap / 穿线 / 出框 / 进零件 / 引线). H_DD and H_TE are ignored
    because production drawings routinely trip those geometry checks.

Writes:
  data/evalsys/annotate_t1/auto_labels.js
  data/evalsys/t1_auto_labels.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.config import load_config  # noqa: E402
from layout_agent.evaluate import Evaluator  # noqa: E402
from layout_agent.models import LayoutState  # noqa: E402

PACK = ROOT / "data" / "evalsys" / "annotate_t1"
MANIFEST = ROOT / "data" / "evalsys" / "pairs_perturb" / "manifest.jsonl"
OUT_JS = PACK / "auto_labels.js"
OUT_JSONL = ROOT / "data" / "evalsys" / "t1_auto_labels.jsonl"

LEVEL_GRADE = {"light": 4, "medium": 3, "heavy": 2}
OP_PROBLEM = {
    "jitter_linear_text": "text_cross_line",
    "shift_dimline": "text_cross_line",
    "swap_layers": "layer_disorder",
    "jitter_radial_text": "leader_too_long",
}
HARD_TO_PROBLEM = {
    "H_TT": "text_overlap",
    "H_TD": "text_cross_line",
    "H_TL": "leader_too_long",
    "H_OOB": "out_of_frame",
    "H_TIN": "text_in_part",
    "H_TOB": "text_cross_line",
}
VISIBLE_HARD = set(HARD_TO_PROBLEM)


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    i = text.find("window.TASKS = ")
    if i < 0:
        raise SystemExit(f"no window.TASKS in {path}")
    raw = text[i + len("window.TASKS = ") :].strip()
    if raw.endswith(";"):
        raw = raw[:-1]
    return json.loads(raw)


def load_manifest_by_state_b(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        sb = rec.get("state_b") or ""
        name = Path(sb).name
        if name:
            out[name] = rec
            out[Path(name).stem] = rec
    return out


def problems_from_ops(ops: List[str]) -> List[str]:
    seen = []
    for op in ops or []:
        key = str(op).split(":", 1)[0]
        tag = OP_PROBLEM.get(key)
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def grade_from_visible(n: int) -> int:
    if n <= 0:
        return 5
    if n <= 2:
        return 4
    if n <= 5:
        return 3
    if n <= 9:
        return 2
    return 1


def label_production(task: Dict[str, Any], ev: Evaluator) -> Optional[Tuple[Dict[str, Any], str]]:
    p = Path(task.get("state") or "")
    if not p.is_file():
        return None
    try:
        st = LayoutState.model_validate(json.loads(p.read_text(encoding="utf-8")))
        report = ev.evaluate(st)
    except Exception as e:  # noqa: BLE001
        print(f"eval fail {p.name}: {e}")
        return None
    codes = Counter(v.code for v in report.hard_violations)
    vis = sum(n for c, n in codes.items() if c in VISIBLE_HARD)
    problems: List[str] = []
    for code, tag in HARD_TO_PROBLEM.items():
        if codes.get(code) and tag not in problems:
            problems.append(tag)
    oc = float(report.metrics.get("S_overcrowd") or 0)
    ly = float(report.metrics.get("S_layer") or 0)
    if oc >= 25 and "overcrowd" not in problems:
        problems.append("overcrowd")
    if ly >= 8 and "layer_disorder" not in problems:
        problems.append("layer_disorder")
    grade = grade_from_visible(vis)
    if vis == 0 and "overcrowd" in problems and grade > 4:
        grade = 4
    if vis == 0 and "layer_disorder" in problems and grade > 4:
        grade = 4
    return (
        {
            "grade": grade,
            "problems": problems,
            "comment": f"auto:prod_vis{vis}",
        },
        f"prod_g{grade}",
    )


def label_task(
    task: Dict[str, Any],
    manifest_b: Dict[str, Dict[str, Any]],
    ev: Optional[Evaluator],
) -> Optional[Tuple[Dict[str, Any], str]]:
    if task.get("source") == "colleague_pairs":
        role = task.get("colleague_role")
        if role == "perfect":
            return (
                {"grade": 5, "problems": [], "comment": "auto:colleague_perfect"},
                "colleague_perfect",
            )
        if role == "perturbed":
            return (
                {"grade": 2, "problems": [], "comment": "auto:colleague_perturbed"},
                "colleague_perturbed",
            )
        return None

    state = str(task.get("state") or "")
    stem = Path(state).stem
    name = Path(state).name
    rec = manifest_b.get(name) or manifest_b.get(stem)
    if rec is None and re.search(r"_(light|medium|heavy)\d+_b\.state", name):
        rec = manifest_b.get(name) or {}
    if rec and ("pairs_perturb" in state.replace("\\", "/") or rec.get("level")):
        level = rec.get("level") or ""
        grade = LEVEL_GRADE.get(level)
        if grade is None:
            return None
        return (
            {
                "grade": grade,
                "problems": problems_from_ops(rec.get("ops") or []),
                "comment": f"auto:perturb_{level}",
            },
            f"perturb_{level}",
        )
    if ev is None:
        return None
    return label_production(task, ev)


def main() -> None:
    tasks = load_tasks(PACK / "tasks.js")
    manifest_b = load_manifest_by_state_b(MANIFEST)
    ev = Evaluator(load_config(ROOT / "config" / "default.yaml"))
    auto: Dict[str, Dict[str, Any]] = {}
    lines: List[str] = []
    counts: Counter[str] = Counter()
    skipped = 0
    for t in tasks:
        got = label_task(t, manifest_b, ev)
        if not got:
            skipped += 1
            continue
        lab, bucket = got
        auto[t["id"]] = lab
        counts[bucket] += 1
        lines.append(
            json.dumps(
                {
                    "sample_id": t["id"],
                    "state": t.get("state"),
                    "view_key": t.get("view_key"),
                    "label": {**lab, "annotator": "auto"},
                },
                ensure_ascii=False,
            )
        )

    js = (
        "window.AUTO_LABEL_VERSION = 2;\n"
        f"window.AUTO_LABELS = {json.dumps(auto, ensure_ascii=False)};\n"
    )
    OUT_JS.write_text(js, encoding="utf-8")
    OUT_JSONL.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"T1 tasks {len(tasks)}")
    print(f"auto-labeled {len(auto)}")
    for k, n in sorted(counts.items()):
        print(f"  {k}: {n}")
    print(f"left unlabeled: {skipped}")
    print(f"wrote {OUT_JS}")
    print(f"wrote {OUT_JSONL}")


if __name__ == "__main__":
    main()
