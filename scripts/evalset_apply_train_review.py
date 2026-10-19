#!/usr/bin/env python3
"""Apply train-set review flags from the local review page.

Reads Chrome notes (or --notes) and:
  swap       -> flip defective/perfect sides, keep in train
  drop       -> remove from train (low-quality / too simple)
  cross_view -> hold out of train into cross_view.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evalset_ingest_human_pairs import (  # noqa: E402
    HUMAN_PAIRS,
    LEGACY_DS,
    NEW_DS,
    REVIEW_ACTIONS,
    relabel_swapped_snapshots,
    swap_pair_sides,
)
from layout_agent.evalset.human_gold import read_jsonl, write_jsonl  # noqa: E402
from layout_agent.evalset.train_variants import write_train_variants  # noqa: E402

NOTES_DEFAULT = HUMAN_PAIRS / "review_notes_chrome.json"
REND = ROOT / "data" / "evalsys" / "review_human_pairs" / "renders"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_notes(path: Path) -> Dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(pid): str(val.get("flag") or "")
        for pid, val in raw.items()
        if isinstance(val, dict) and val.get("flag")
    }


def row(kind: str, rec: Dict[str, Any]) -> Dict[str, str]:
    return {
        "pair_id": str(rec["pair_id"]),
        "view_key": str(rec["view_key"]),
        "source": str(rec.get("source") or ""),
        "action": kind,
    }


def write_actions(train: List[Dict[str, Any]], test: List[Dict[str, Any]], notes: Dict[str, str]) -> Dict[str, Any]:
    by_id = {r["pair_id"]: r for r in train}
    test_by_id = {r["pair_id"]: r for r in test}
    swap, drop, cross = [], [], []
    skipped_test = []
    unknown = []
    for pid, flag in notes.items():
        rec = by_id.get(pid)
        if rec is None:
            if pid in test_by_id:
                skipped_test.append({"pair_id": pid, "flag": flag, "split": "test"})
            else:
                unknown.append({"pair_id": pid, "flag": flag})
            continue
        if flag == "swap":
            swap.append(row("swap", rec))
        elif flag == "drop":
            drop.append(row("drop", rec))
        elif flag == "bad":
            cross.append(row("cross_view", rec))
    # Keep previously confirmed test swaps in the durable list.
    for rec in test:
        if rec.get("sides_swapped"):
            item = row("swap", rec)
            if item["view_key"] not in {r["view_key"] for r in swap}:
                item["split"] = "test"
                swap.append(item)
    actions = {
        "reviewed_at": _now(),
        "source_notes": str(NOTES_DEFAULT.as_posix()),
        "counts": {
            "swap_train": sum(1 for r in swap if r.get("split") != "test"),
            "swap_total": len(swap),
            "drop": len(drop),
            "cross_view": len(cross),
        },
        "swap": swap,
        "drop": drop,
        "cross_view": cross,
        "skipped_test_flags": skipped_test,
        "unknown_ids": unknown,
    }
    REVIEW_ACTIONS.write_text(json.dumps(actions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return actions


def patch_manifest(path: Path, swap_ids: set[str]) -> int:
    if not path.exists():
        return 0
    man = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for rec in man.get("pairs") or []:
        pid = str(rec.get("pair_id") or "")
        vkey = str(rec.get("view_key") or "")
        if rec.get("sides_swapped"):
            continue
        if pid in swap_ids or vkey in swap_ids:
            rec.update(swap_pair_sides(rec))
            n += 1
    path.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return n


def drop_renders(pair_ids: List[str]) -> None:
    if not REND.exists():
        return
    for pid in pair_ids:
        for side in ("a", "b"):
            p = REND / f"{pid}_{side}.png"
            if p.exists():
                p.unlink()


def main() -> None:
    notes_path = NOTES_DEFAULT
    if not notes_path.exists():
        raise SystemExit(f"missing notes: {notes_path}")
    notes = load_notes(notes_path)
    train = read_jsonl(HUMAN_PAIRS / "train.jsonl")
    test = read_jsonl(HUMAN_PAIRS / "test.jsonl")
    dropped = read_jsonl(HUMAN_PAIRS / "dropped.jsonl")
    actions = write_actions(train, test, notes)
    print("actions", actions["counts"])

    swap_ids = {r["pair_id"] for r in actions["swap"] if r.get("split") != "test"}
    swap_keys = {r["view_key"] for r in actions["swap"]}
    drop_ids = {r["pair_id"] for r in actions["drop"]}
    drop_keys = {r["view_key"] for r in actions["drop"]}
    cross_ids = {r["pair_id"] for r in actions["cross_view"]}
    cross_keys = {r["view_key"] for r in actions["cross_view"]}

    new_train: List[Dict[str, Any]] = []
    new_drop: List[Dict[str, Any]] = []
    cross_rows: List[Dict[str, Any]] = []
    swapped: List[str] = []
    for rec in train:
        pid = rec["pair_id"]
        if pid in drop_ids:
            item = dict(rec)
            item["reason"] = "review_drop_low_quality"
            item["review_flag"] = "drop"
            new_drop.append(item)
            continue
        if pid in cross_ids:
            item = dict(rec)
            item["held_out"] = "cross_view"
            item["reason"] = "cross_view_annotation_mismatch"
            item["split"] = "cross_view"
            item["review_flag"] = "bad"
            cross_rows.append(item)
            continue
        if pid in swap_ids and not rec.get("sides_swapped"):
            rec = swap_pair_sides(rec)
            relabel_swapped_snapshots(rec)
            swapped.append(pid)
        new_train.append(rec)

    existing_drop_keys = {str(r.get("view_key") or "") for r in dropped}
    for rec in new_drop:
        if rec["view_key"] not in existing_drop_keys:
            dropped.append(rec)

    write_jsonl(HUMAN_PAIRS / "train.jsonl", new_train)
    write_jsonl(HUMAN_PAIRS / "dropped.jsonl", dropped)
    write_jsonl(HUMAN_PAIRS / "cross_view.jsonl", cross_rows)
    write_jsonl(HUMAN_PAIRS / "all_modified.jsonl", test + new_train)

    n_ent = patch_manifest(NEW_DS / "manifest.json", swap_ids | swap_keys)
    n_leg = patch_manifest(LEGACY_DS / "manifest.json", swap_ids | swap_keys)
    print(f"swapped train={len(swapped)} manifest_ent={n_ent} manifest_legacy={n_leg}")

    drop_renders(sorted(drop_ids | cross_ids | set(swapped)))

    registry_path = HUMAN_PAIRS / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["counts"]["modified_pairs"] = len(test) + len(new_train)
    registry["counts"]["train_pairs"] = len(new_train)
    registry["counts"]["test_pairs"] = len(test)
    registry["counts"]["dropped"] = len(dropped)
    registry["counts"]["cross_view"] = len(cross_rows)
    registry["counts"]["train_solutions"] = len({r["solution_id"] for r in new_train})
    registry["train_view_keys"] = sorted({r["view_key"] for r in new_train})
    registry["cross_view_view_keys"] = sorted(cross_keys)
    registry["train_review"] = {
        "reviewed_at": actions["reviewed_at"],
        "swap": actions["counts"]["swap_train"],
        "drop": len(drop_ids),
        "cross_view": len(cross_ids),
        "train_pairs_after": len(new_train),
    }
    corrections = list(registry.get("label_corrections") or [])
    have = {c.get("pair_id") for c in corrections}
    for rec in new_train:
        if rec.get("sides_swapped") and rec["pair_id"] not in have:
            corrections.append(
                {
                    "view_key": rec["view_key"],
                    "pair_id": rec["pair_id"],
                    "action": "swap_a_b",
                    "reason": "annotator reversed left/right; human-reviewed 2026-09-17",
                }
            )
    registry["label_corrections"] = corrections
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    synth_stats = write_train_variants(ROOT)
    print("train_variants", synth_stats)
    print(
        json.dumps(
            {
                "train": len(new_train),
                "test": len(test),
                "dropped_added": len(new_drop),
                "dropped_total": len(dropped),
                "cross_view": len(cross_rows),
                "swapped": swapped,
                "flags": Counter(notes.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
