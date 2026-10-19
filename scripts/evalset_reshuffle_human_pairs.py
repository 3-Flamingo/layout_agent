#!/usr/bin/env python3
"""Re-split real human_pairs train+test by uniform random sample.

Does not put dropped / cross_view / CMU synth into test. After writing the new
freeze, rebuilds train_with_synth (dropping any synth whose view is now test).

  python scripts/evalset_reshuffle_human_pairs.py
  python scripts/evalset_review_human_pairs.py --skip-existing
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.evalset.human_gold import read_jsonl, write_jsonl  # noqa: E402
from layout_agent.evalset.isolation import (  # noqa: E402
    forbidden_training_views,
    gold_views,
    write_forbidden_train_views,
)
from layout_agent.evalset.split_human_pairs import (  # noqa: E402
    SPLIT_METHOD,
    SPLIT_N_TEST,
    SPLIT_SEED,
    assign_splits,
    random_test_keys,
)
from layout_agent.evalset.train_variants import write_train_variants  # noqa: E402

HP = ROOT / "data" / "evalsys" / "human_pairs"
GOLD_DIR = ROOT / "data" / "evalsys" / "human_gold"


def _delta_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    vals = sorted(int(r.get("pose_delta") or 0) for r in rows)
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": vals[0],
        "median": vals[len(vals) // 2],
        "mean": round(sum(vals) / len(vals), 2),
        "max": vals[-1],
    }


def main() -> None:
    train = read_jsonl(HP / "train.jsonl")
    test = read_jsonl(HP / "test.jsonl")
    pool = train + test
    gold = gold_views(GOLD_DIR)
    pool = [rec for rec in pool if str(rec.get("view_key") or "") not in gold]
    keys = [str(rec.get("view_key") or "") for rec in pool]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate view_key in train+test pool")
    test_keys = set(random_test_keys(keys, n_test=SPLIT_N_TEST, seed=SPLIT_SEED))
    test_rows, train_rows = assign_splits(pool, test_keys)
    if len(test_rows) != SPLIT_N_TEST:
        raise SystemExit(f"expected {SPLIT_N_TEST} test rows, got {len(test_rows)}")

    write_jsonl(HP / "test.jsonl", test_rows)
    write_jsonl(HP / "train.jsonl", train_rows)
    write_jsonl(HP / "all_modified.jsonl", test_rows + train_rows)

    now = datetime.now(timezone.utc).astimezone().isoformat()
    registry_path = HP / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    old_test = list(registry.get("test_view_keys") or [])
    if old_test and "prior_quality_test_view_keys" not in registry:
        registry["prior_quality_test_view_keys"] = old_test
    policy = dict(registry.get("policy") or {})
    policy.update(
        {
            "test_is_frozen": True,
            "new_pairs_go_to": "train",
            "skip_if_view_key_in_test": True,
            "split_method": SPLIT_METHOD,
            "seed": SPLIT_SEED,
            "n_test": SPLIT_N_TEST,
            "label": "prefer perfect/right (b) over defective/left (a)",
        }
    )
    registry["policy"] = policy
    registry["frozen_at"] = now
    counts = dict(registry.get("counts") or {})
    counts["modified_pairs"] = len(pool)
    counts["test_pairs"] = len(test_rows)
    counts["train_pairs"] = len(train_rows)
    counts["test_solutions"] = len({r.get("solution_id") for r in test_rows})
    counts["train_solutions"] = len({r.get("solution_id") for r in train_rows})
    registry["counts"] = counts
    registry["test_view_keys"] = sorted(test_keys)
    registry["train_view_keys"] = sorted({r["view_key"] for r in train_rows})
    registry["split_stats"] = {
        "method": SPLIT_METHOD,
        "seed": SPLIT_SEED,
        "test_pose_delta": _delta_stats(test_rows),
        "train_pose_delta": _delta_stats(train_rows),
        "pool_pose_delta": _delta_stats(pool),
    }
    registry["how_to_append"] = (
        "python scripts/evalset_ingest_human_pairs.py --append-only "
        "# new modified pairs go to train.jsonl; test.jsonl stays frozen"
    )
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    forbidden = forbidden_training_views(ROOT)
    write_forbidden_train_views(ROOT, forbidden)
    synth = write_train_variants(ROOT)

    moved_to_test = sorted(set(old_test) - test_keys) if old_test else []
    moved_to_train = sorted(test_keys - set(old_test)) if old_test else sorted(test_keys)
    print(
        json.dumps(
            {
                "test": len(test_rows),
                "train": len(train_rows),
                "test_pose_delta": _delta_stats(test_rows),
                "train_pose_delta": _delta_stats(train_rows),
                "gold_overlap_test": len(test_keys & gold),
                "synth": synth,
                "views_leaving_old_test": len(moved_to_test),
                "views_entering_new_test": len(moved_to_train),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
