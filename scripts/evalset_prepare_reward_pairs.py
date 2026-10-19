#!/usr/bin/env python3
"""Render ICM for human_pairs train.jsonl and write a reward-model manifest.

Test / gold / holdout view_keys are excluded. A 15% view-hashed val split is
carved from train only (same view never crosses train/val).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evalset_perturb_pairs import adaptive_pixel_size  # noqa: E402
from layout_agent.config import load_config  # noqa: E402
from layout_agent.evalset.human_gold import read_jsonl, write_jsonl  # noqa: E402
from layout_agent.evalset.isolation import forbidden_training_views  # noqa: E402
from layout_agent.evalset.pairs import split_for  # noqa: E402
from layout_agent.models import LayoutState  # noqa: E402
from layout_agent.perceive import Perceiver  # noqa: E402

HP = ROOT / "data" / "evalsys" / "human_pairs"
OUT = ROOT / "data" / "evalsys" / "reward_human_pairs"


def _rel(out: Path, path: Path) -> str:
    if not path.is_absolute():
        path = ROOT / path
    return Path(os.path.relpath(path.resolve(), out.resolve())).as_posix()


def _load_state(rel: str) -> LayoutState:
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    return LayoutState.model_validate(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HP / "train.jsonl"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT / src
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    icm_dir = out / "icm"
    icm_dir.mkdir(parents=True, exist_ok=True)

    blocked = forbidden_training_views(ROOT)
    rows = read_jsonl(src)
    cfg = load_config(ROOT / "config" / "default.yaml")
    perceiver = Perceiver(cfg)
    base_ps = float(perceiver.pixel_size)
    kept: List[Dict[str, Any]] = []
    n_block = n_fail = 0
    for rec in rows:
        vkey = str(rec.get("view_key") or "")
        if not vkey or vkey in blocked:
            n_block += 1
            continue
        pid = str(rec["pair_id"])
        try:
            state_a = _load_state(str(rec["state_a"]))
            state_b = _load_state(str(rec["state_b"]))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {pid}: parse {exc}")
            n_fail += 1
            continue
        item = dict(rec)
        item["split"] = split_for(vkey)
        for which, state in (("a", state_a), ("b", state_b)):
            inst = icm_dir / f"{pid}_{which}.instance.png"
            obs = icm_dir / f"{pid}_{which}.obstacle.png"
            if not (args.skip_existing and inst.exists() and obs.exists()):
                try:
                    perceiver.pixel_size = adaptive_pixel_size(state, base_ps)
                    perceiver.render_icm(state, inst, obs)
                except Exception as exc:  # noqa: BLE001
                    print(f"skip {pid} {which}: render {exc}")
                    n_fail += 1
                    inst = None
                    break
            if inst is None:
                break
            item[f"icm_{which}"] = _rel(out, inst)
            item[f"state_{which}"] = _rel(out, ROOT / rec[f"state_{which}"])
        if not item.get("icm_a") or not item.get("icm_b"):
            continue
        prefer = str((item.get("label") or {}).get("prefer") or "").lower()
        if prefer not in ("a", "b"):
            item["label"] = {"prefer": "b"}
        kept.append(item)
        if len(kept) % 50 == 0:
            print(f"rendered {len(kept)}")

    write_jsonl(out / "manifest.jsonl", kept)
    n_train = sum(1 for r in kept if r.get("split") == "train")
    n_val = sum(1 for r in kept if r.get("split") == "val")
    print(
        f"wrote {out / 'manifest.jsonl'} n={len(kept)} train={n_train} val={n_val} "
        f"blocked={n_block} fail={n_fail}"
    )


if __name__ == "__main__":
    main()
