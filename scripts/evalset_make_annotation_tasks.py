#!/usr/bin/env python3
"""Build an annotation package (tasks.js + annotate.html + renders).

Two modes:
  prefer: sample pairs from a pairs manifest.jsonl (T2 / R3 labeling)
  grade:  render previews for a set of LayoutState files (T1 labeling)

The output folder is self-contained: open annotate.html in a browser,
label, then export labels .jsonl (format matches evalset_baseline_consistency).

Usage:
  python scripts/evalset_make_annotation_tasks.py prefer \
      --manifest data/evalsys/pairs_perturb/manifest.jsonl \
      --out data/evalsys/annotate_t2 --n 200 --seed 7
  python scripts/evalset_make_annotation_tasks.py grade \
      --states "data/datasets/*/snapshots/*.state.json" \
      --out data/evalsys/annotate_t1 --n 100 --seed 7
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.evalset import render_preview, view_key
from layout_agent.models import LayoutState


def _write_package(out: Path, tasks: List[Dict], dataset_id: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "tools" / "annotate.html", out / "annotate.html")
    js = (
        f"window.DATASET_ID = {json.dumps(dataset_id)};\n"
        f"window.TASKS = {json.dumps(tasks, ensure_ascii=False)};\n"
    )
    (out / "tasks.js").write_text(js, encoding="utf-8")
    print(f"package: {out}\\annotate.html  ({len(tasks)} tasks)")


def mode_prefer(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    pairs_root = manifest.parent
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    rng = random.Random(args.seed)

    records = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.n and len(records) > args.n:
        records = rng.sample(records, args.n)

    rend = out / "renders"
    rend.mkdir(parents=True, exist_ok=True)
    tasks = []
    for rec in records:
        pid = rec["pair_id"]
        # randomize left/right so annotators can't learn "a is always original"
        flip = rng.random() < 0.5
        ra, rb = rec["render_a"], rec["render_b"]
        sa, sb = rec["state_a"], rec["state_b"]
        if flip:
            ra, rb, sa, sb = rb, ra, sb, sa
        for src, dst in ((ra, f"{pid}_L.png"), (rb, f"{pid}_R.png")):
            shutil.copy(pairs_root / src, rend / dst)
        tasks.append(
            {
                "id": pid,
                "type": "prefer",
                "view_key": rec.get("view_key", ""),
                "render_a": f"renders/{pid}_L.png",
                "render_b": f"renders/{pid}_R.png",
                # state paths follow the displayed order so exported labels stay correct
                "state_a": str((pairs_root / sa).resolve().as_posix()),
                "state_b": str((pairs_root / sb).resolve().as_posix()),
            }
        )
    _write_package(out, tasks, out.name)


def mode_grade(args: argparse.Namespace) -> None:
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    rng = random.Random(args.seed)
    paths = [Path(p) for p in glob.glob(args.states, recursive=True)]
    # dedupe by view_key
    seen = set()
    uniq: List[Path] = []
    for p in sorted(paths):
        try:
            st = LayoutState.model_validate(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
        vk = view_key(st)
        if vk in seen:
            continue
        seen.add(vk)
        uniq.append(p)
    if args.n and len(uniq) > args.n:
        uniq = rng.sample(uniq, args.n)

    rend = out / "renders"
    rend.mkdir(parents=True, exist_ok=True)
    tasks = []
    for p in uniq:
        st = LayoutState.model_validate(json.loads(p.read_text(encoding="utf-8")))
        sid = st.state_id or p.stem
        try:
            render_preview(st, rend / f"{sid}.png")
        except Exception as e:  # noqa: BLE001
            print(f"render failed {p.name}: {e}")
            continue
        tasks.append(
            {
                "id": sid,
                "type": "grade",
                "render": f"renders/{sid}.png",
                "state": str(p.resolve().as_posix()),
            }
        )
    _write_package(out, tasks, out.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p1 = sub.add_parser("prefer")
    p1.add_argument("--manifest", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--n", type=int, default=0)
    p1.add_argument("--seed", type=int, default=7)
    p2 = sub.add_parser("grade")
    p2.add_argument("--states", required=True, help="glob, e.g. data/datasets/*/snapshots/*.state.json")
    p2.add_argument("--out", required=True)
    p2.add_argument("--n", type=int, default=0)
    p2.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.mode == "prefer":
        mode_prefer(args)
    else:
        mode_grade(args)


if __name__ == "__main__":
    main()
