"""View isolation for gold holdout and frozen human-pair test views.

Gold view_keys and the frozen human-pair test split never enter training or
checkpoint selection. T2 gold may be a subset of human_pairs test; those views
are already forbidden. Collectors, PPO and rollout eval must use the same set.
Legacy colleague_challenge.jsonl is used only when human_pairs/test.jsonl is
absent (pre-ingest fallback).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Set


def read_view_lines(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def jsonl_view_keys(path: Path, field: str = "view_key") -> Set[str]:
    keys: Set[str] = set()
    if not path.exists():
        return keys
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key = str(json.loads(line).get(field) or "")
        if key:
            keys.add(key)
    return keys


def colleague_challenge_path(root: Path) -> Path:
    return Path(root) / "data" / "evalsys" / "proxy" / "colleague_challenge.jsonl"


def colleague_challenge_views(root: Path) -> Set[str]:
    return jsonl_view_keys(colleague_challenge_path(root))


def human_pairs_test_path(root: Path) -> Path:
    return Path(root) / "data" / "evalsys" / "human_pairs" / "test.jsonl"


def human_pairs_test_views(root: Path) -> Set[str]:
    return jsonl_view_keys(human_pairs_test_path(root))


def human_pairs_train_path(root: Path) -> Path:
    return Path(root) / "data" / "evalsys" / "human_pairs" / "train.jsonl"


def human_pairs_train_views(root: Path) -> Set[str]:
    return jsonl_view_keys(human_pairs_train_path(root))


def gold_views(gold_dir: Path) -> Set[str]:
    keys: Set[str] = set()
    for name in ("t1_tasks.jsonl", "t2_tasks.jsonl"):
        keys |= jsonl_view_keys(gold_dir / name)
    return keys


def forbidden_training_views(
    root: Path,
    *,
    holdout: Optional[Path] = None,
    gold_dir: Optional[Path] = None,
    include_colleague: bool = True,
) -> Set[str]:
    evalsys = Path(root) / "data" / "evalsys"
    keys = read_view_lines(holdout or evalsys / "holdout_views.txt")
    keys |= gold_views(gold_dir or evalsys / "human_gold")
    if include_colleague:
        test_views = human_pairs_test_views(root)
        if test_views:
            keys |= test_views
        else:
            keys |= colleague_challenge_views(root)
    return {key for key in keys if key}


def write_forbidden_train_views(root: Path, keys: Iterable[str]) -> Path:
    path = Path(root) / "data" / "evalsys" / "forbidden_train_views.txt"
    unique = sorted(set(str(key) for key in keys if key))
    header = (
        "# Never train or select checkpoints on these view_keys.\n"
        "# Union of perturb holdout, frozen human gold, and human-pair test views.\n"
    )
    path.write_text(header + "\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")
    return path
