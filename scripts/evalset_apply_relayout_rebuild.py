#!/usr/bin/env python3
"""Apply relayout review flags, synthesize CMU pairs, then randomly resplit.

1. Dump Chrome notes and apply them to the 707 relayout pairs:
     drop -> deleted from datasets
     swap -> flip defective/perfect
     bad  -> cross_view.jsonl only (no train/test/synth)
     unlabeled after a completed review pass -> keep
2. CMU-style perturb of complex perfect sides, count ≈ kept real pairs.
3. Test is a uniform random sample of *real* pairs. Test size is 20% of
   (real + synth). Remaining real (+ non-test-view synth) is train.

Old easy-initial-state train/test files are archived, not mixed in.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evalset_cmu_perturb_pairs import generate_cmu_pairs  # noqa: E402
from evalset_dump_review_notes import OUT as NOTES_OUT  # noqa: E402
from evalset_dump_review_notes import main as dump_notes  # noqa: E402
from evalset_ingest_human_pairs import (  # noqa: E402
    HUMAN_PAIRS,
    relabel_swapped_snapshots,
    swap_pair_sides,
)
from layout_agent.evalset.human_gold import read_jsonl, write_jsonl  # noqa: E402
from layout_agent.evalset.isolation import (  # noqa: E402
    forbidden_training_views,
    gold_views,
    write_forbidden_train_views,
)
from layout_agent.evalset.split_human_pairs import (  # noqa: E402
    SPLIT_METHOD,
    SPLIT_SEED,
    SPLIT_TEST_RATIO,
    assign_splits,
    n_test_from_all,
    random_test_keys,
)
from layout_agent.evalset.train_variants import write_train_variants  # noqa: E402

GOLD_DIR = ROOT / "data" / "evalsys" / "human_gold"
REND = ROOT / "data" / "evalsys" / "review_human_pairs" / "renders"
ARCHIVE = HUMAN_PAIRS / "archive_pre_relayout"
ACTIONS = HUMAN_PAIRS / "review_actions_relayout.json"
CMU_SEED = 202609182


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_notes(path: Path) -> Dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(pid): str(val.get("flag") or "")
        for pid, val in raw.items()
        if isinstance(val, dict) and val.get("flag")
    }


def flag_of(rec: Dict[str, Any], notes: Dict[str, str]) -> str:
    pid = str(rec["pair_id"])
    if notes.get(pid):
        return notes[pid]
    return "ok"


def apply_flags(
    rows: List[Dict[str, Any]], notes: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    cross: List[Dict[str, Any]] = []
    swapped: List[str] = []
    for rec in rows:
        flag = flag_of(rec, notes)
        item = dict(rec)
        item["review_flag"] = flag
        if flag == "drop":
            item["reason"] = "review_drop_low_quality"
            item["split"] = "dropped"
            dropped.append(item)
            continue
        if flag == "bad":
            item["held_out"] = "cross_view"
            item["reason"] = "cross_view_annotation_mismatch"
            item["split"] = "cross_view"
            cross.append(item)
            continue
        if flag == "swap" and not item.get("sides_swapped"):
            item = swap_pair_sides(item)
            item["review_flag"] = "swap"
            relabel_swapped_snapshots(item)
            swapped.append(str(item["pair_id"]))
        item["split"] = "real"
        kept.append(item)
    return kept, dropped, cross, swapped


def drop_renders(pair_ids: Set[str]) -> None:
    if not REND.exists():
        return
    for pid in pair_ids:
        for side in ("a", "b"):
            p = REND / f"{pid}_{side}.png"
            if p.exists():
                p.unlink()


def archive_previous() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for name in (
        "train.jsonl",
        "test.jsonl",
        "train_with_synth.jsonl",
        "all_modified.jsonl",
        "cmu_perturb.jsonl",
        "registry.json",
    ):
        src = HUMAN_PAIRS / name
        if src.exists():
            shutil.copy2(src, ARCHIVE / name)


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


def write_registry(
    *,
    kept: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    train_rows: List[Dict[str, Any]],
    dropped: List[Dict[str, Any]],
    cross: List[Dict[str, Any]],
    synth: List[Dict[str, Any]],
    notes: Dict[str, str],
    swapped: List[str],
    flag_counts: Dict[str, int],
    n_test: int,
    cmu_meta: Dict[str, Any],
    unlabeled_kept: int,
) -> None:
    path = HUMAN_PAIRS / "registry.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if old.get("test_view_keys") and "prior_random_test_view_keys" not in old:
        old["prior_random_test_view_keys"] = list(old.get("test_view_keys") or [])
    old["frozen_at"] = _now()
    old["policy"] = {
        "test_is_frozen": True,
        "test_source": "real_only",
        "test_ratio_of_all": SPLIT_TEST_RATIO,
        "split_method": SPLIT_METHOD,
        "seed": SPLIT_SEED,
        "n_test": n_test,
        "drop_identical_sides": True,
        "synth_ratio_target": 1.0,
        "label": "prefer perfect/right (b) over defective/left (a)",
        "real_batch": "colleague_enterprise_t3_relayout",
        "old_easy_initial_states": "archived to data/evalsys/human_pairs/archive_pre_relayout",
    }
    old["counts"] = {
        "modified_pairs": len(kept),
        "test_pairs": len(test_rows),
        "train_pairs": len(train_rows),
        "train_pairs_real": len(train_rows),
        "dropped": len(dropped),
        "cross_view": len(cross),
        "cmu_perturb": len(synth),
        "test_solutions": len({r.get("solution_id") for r in test_rows}),
        "train_solutions": len({r.get("solution_id") for r in train_rows}),
    }
    old["test_view_keys"] = sorted({r["view_key"] for r in test_rows})
    old["train_view_keys"] = sorted({r["view_key"] for r in train_rows})
    old["cross_view_view_keys"] = sorted({str(r.get("view_key") or "") for r in cross if r.get("view_key")})
    old["split_stats"] = {
        "method": SPLIT_METHOD,
        "seed": SPLIT_SEED,
        "n_test": n_test,
        "ratio_of_all": SPLIT_TEST_RATIO,
        "n_real": len(kept),
        "n_synth": len(synth),
        "n_all_for_ratio": len(kept) + len(synth),
        "test_pose_delta": _delta_stats(test_rows),
        "train_pose_delta": _delta_stats(train_rows),
        "pool_pose_delta": _delta_stats(kept),
    }
    old["relayout_review"] = {
        "reviewed_at": _now(),
        "notes": str(NOTES_OUT.as_posix()),
        "n_relayout": 707,
        "flags": flag_counts,
        "unlabeled_kept_as_ok": unlabeled_kept,
        "swap": len(swapped),
        "drop": len(dropped),
        "cross_view_new": sum(1 for r in cross if str(r.get("pair_id") or "").startswith("relayout_")),
        "kept_real": len(kept),
        "cmu": cmu_meta,
    }
    old["how_to_append"] = (
        "python scripts/evalset_ingest_relayout_review.py  "
        "# review-only; do not mix into this freeze until a new random split"
    )
    path.write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    dump_notes()
    notes = load_notes(NOTES_OUT)
    relayout = read_jsonl(HUMAN_PAIRS / "relayout.jsonl")
    if len(relayout) != 707:
        print(f"WARN relayout.jsonl has {len(relayout)} rows, expected 707")

    relayout_ids = {str(r["pair_id"]) for r in relayout}
    relayout_notes = {k: v for k, v in notes.items() if k in relayout_ids}
    unlabeled = sorted(relayout_ids - set(relayout_notes))
    flag_counts = dict(Counter(relayout_notes.values()))
    flag_counts["unlabeled_kept"] = len(unlabeled)
    print("relayout flags", flag_counts)

    kept, new_drop, new_cross, swapped = apply_flags(relayout, relayout_notes)
    gold = gold_views(GOLD_DIR)
    n_gold = sum(1 for r in kept if str(r.get("view_key") or "") in gold)
    if n_gold:
        print(f"exclude {n_gold} gold-overlapping views from real pool")
        kept = [r for r in kept if str(r.get("view_key") or "") not in gold]

    archive_previous()

    prior_drop = read_jsonl(HUMAN_PAIRS / "dropped.jsonl") if (HUMAN_PAIRS / "dropped.jsonl").exists() else []
    drop_ids = {str(r.get("pair_id") or "") for r in prior_drop}
    dropped = list(prior_drop)
    for rec in new_drop:
        if rec.get("pair_id") not in drop_ids:
            dropped.append(rec)
            drop_ids.add(str(rec.get("pair_id") or ""))

    prior_cross = read_jsonl(HUMAN_PAIRS / "cross_view.jsonl") if (HUMAN_PAIRS / "cross_view.jsonl").exists() else []
    cross_ids = {str(r.get("pair_id") or "") for r in prior_cross}
    cross = list(prior_cross)
    for rec in new_cross:
        if rec.get("pair_id") not in cross_ids:
            cross.append(rec)
            cross_ids.add(str(rec.get("pair_id") or ""))

    blocked_synth: Set[str] = set(gold) | {
        str(r.get("view_key") or "") for r in dropped if r.get("view_key")
    } | {
        str(r.get("view_key") or "") for r in cross if r.get("view_key")
    }
    cap = len(kept)
    synth, cmu_meta = generate_cmu_pairs(
        kept,
        cap=cap,
        seed=CMU_SEED,
        min_ann=4,
        min_delta=1,
        blocked=blocked_synth,
        render=False,
        write_files=True,
    )

    n_test = n_test_from_all(len(kept), len(synth), ratio=SPLIT_TEST_RATIO)
    keys = [str(r.get("view_key") or "") for r in kept]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate view_key in kept real pool")
    test_keys = set(random_test_keys(keys, n_test=n_test, seed=SPLIT_SEED))
    test_rows, train_rows = assign_splits(kept, test_keys)
    if len(test_rows) != n_test:
        raise SystemExit(f"expected {n_test} test rows, got {len(test_rows)}")

    write_jsonl(HUMAN_PAIRS / "test.jsonl", test_rows)
    write_jsonl(HUMAN_PAIRS / "train.jsonl", train_rows)
    write_jsonl(HUMAN_PAIRS / "all_modified.jsonl", test_rows + train_rows)
    write_jsonl(HUMAN_PAIRS / "dropped.jsonl", dropped)
    write_jsonl(HUMAN_PAIRS / "cross_view.jsonl", cross)
    write_jsonl(HUMAN_PAIRS / "relayout_kept.jsonl", kept)

    ACTIONS.write_text(
        json.dumps(
            {
                "reviewed_at": _now(),
                "source_notes": str(NOTES_OUT.as_posix()),
                "counts": {
                    "keep": len(kept),
                    "drop": len(new_drop),
                    "swap": len(swapped),
                    "cross_view": len(new_cross),
                    "unlabeled_kept": len(unlabeled),
                },
                "swap": [{"pair_id": pid} for pid in swapped],
                "drop": [{"pair_id": r["pair_id"], "view_key": r.get("view_key")} for r in new_drop],
                "cross_view": [{"pair_id": r["pair_id"], "view_key": r.get("view_key")} for r in new_cross],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    drop_renders(set(swapped))
    write_registry(
        kept=kept,
        test_rows=test_rows,
        train_rows=train_rows,
        dropped=dropped,
        cross=cross,
        synth=synth,
        notes=relayout_notes,
        swapped=swapped,
        flag_counts=flag_counts,
        n_test=n_test,
        cmu_meta={k: cmu_meta.get(k) for k in ("seed", "cap", "kept", "candidates", "skips", "min_ann")},
        unlabeled_kept=len(unlabeled),
    )
    forbidden = forbidden_training_views(ROOT)
    write_forbidden_train_views(ROOT, forbidden)
    variants = write_train_variants(ROOT)
    print(
        json.dumps(
            {
                "keep_real": len(kept),
                "drop_new": len(new_drop),
                "swap": len(swapped),
                "cross_view_new": len(new_cross),
                "unlabeled_kept": len(unlabeled),
                "synth": len(synth),
                "n_test": n_test,
                "test": len(test_rows),
                "train_real": len(train_rows),
                "train_variants": variants,
                "test_pose_delta": _delta_stats(test_rows),
                "train_pose_delta": _delta_stats(train_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
