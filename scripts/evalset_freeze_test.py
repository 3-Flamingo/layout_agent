#!/usr/bin/env python3
"""Freeze the holdout view_key list from the perturb-pair val split.

Writes:
  data/evalsys/holdout_views.txt
  data/evalsys/proxy_holdout_manifest.json

The 80-T2 / 15-T1 public gold protocol is owned by
scripts/evalset_human_gold.py; this script must not overwrite it.

Does not copy training views into the test packages.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "evalsys" / "pairs_perturb" / "manifest.jsonl"
OUT_HOLDOUT = ROOT / "data" / "evalsys" / "holdout_views.txt"
OUT_META = ROOT / "data" / "evalsys" / "proxy_holdout_manifest.json"


def main() -> None:
    records = [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]
    val = [r for r in records if r.get("split") == "val"]
    train = [r for r in records if r.get("split") != "val"]
    val_views = sorted({r["view_key"] for r in val})
    train_views = {r["view_key"] for r in train}
    overlap = set(val_views) & train_views
    if overlap:
        print(f"ERROR: train/val view_key overlap: {overlap}", file=sys.stderr)
        sys.exit(1)

    levels = Counter(r.get("level", "?") for r in val)
    OUT_HOLDOUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_HOLDOUT.write_text(
        "# Frozen holdout view_keys (perturb pairs split=val). Do not train on these.\n"
        + "\n".join(val_views)
        + "\n",
        encoding="utf-8",
    )
    meta = {
        "frozen_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source_pairs_manifest": str(MANIFEST.relative_to(ROOT).as_posix()),
        "protocol": {
            "t2": "human pairwise prefer on val pairs; left/right randomized",
            "t1": "human 1-5 grade on holdout original + perturbed layouts",
            "split_rule": "hash(view_key) val_ratio=0.15; same view never crosses train/val",
        },
        "holdout_views": val_views,
        "n_holdout_views": len(val_views),
        "n_t2_pairs": len(val),
        "t2_level_counts": dict(levels),
        "n_train_pairs_excluded": len(train),
        "n_train_views_excluded": len(train_views),
        "packages": {
            "t1": "data/evalsys/annotate_t1/",
            "t2": "data/evalsys/annotate_t2/",
        },
        "labels": {
            "t1": None,
            "t2": None,
        },
        "status": "packages_pending_or_ready; labels not yet frozen",
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"holdout views: {len(val_views)} -> {OUT_HOLDOUT}")
    print(f"T2 val pairs: {len(val)} levels={dict(levels)}")
    print(f"train excluded: {len(train)} pairs / {len(train_views)} views")
    print(f"meta: {OUT_META}")


if __name__ == "__main__":
    main()
