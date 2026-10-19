#!/usr/bin/env python3
"""Freeze reproducible weak-supervision proxy sets.

Outputs:
  data/evalsys/proxy/perturb_val.jsonl
  data/evalsys/proxy/colleague_challenge.jsonl
  data/evalsys/proxy/manifest.json

The script also removes duplicate T1 task/sample IDs by keeping the last
occurrence. Annotation renders are keyed by task ID and later package writes
overwrite earlier renders, so the last task is the one matching the image.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
EVALSYS = ROOT / "data" / "evalsys"
T1_TASKS = EVALSYS / "annotate_t1" / "tasks.js"
T1_LABELS = EVALSYS / "t1_auto_labels.jsonl"
T2_TASKS = EVALSYS / "annotate_t2" / "tasks.js"
PAIR_MANIFEST = EVALSYS / "pairs_perturb" / "manifest.jsonl"
OUT = EVALSYS / "proxy"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tasks_js(path: Path) -> Tuple[List[Dict[str, Any]], str, int, int]:
    text = path.read_text(encoding="utf-8")
    marker = "window.TASKS = "
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"no {marker!r} in {path}")
    json_start = start + len(marker)
    tasks, consumed = json.JSONDecoder().raw_decode(text[json_start:])
    if not isinstance(tasks, list):
        raise ValueError(f"window.TASKS is not an array in {path}")
    return tasks, text, json_start, json_start + consumed


def write_tasks_js(
    path: Path,
    tasks: List[Dict[str, Any]],
    text: str,
    json_start: int,
    json_end: int,
) -> None:
    encoded = json.dumps(tasks, ensure_ascii=False)
    path.write_text(text[:json_start] + encoded + text[json_end:], encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    rows = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def dedupe_keep_last(
    records: List[Dict[str, Any]], key: str
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    positions: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        positions[str(rec.get(key) or "")].append(i)
    dupes = {
        ident: [records[i] for i in indexes]
        for ident, indexes in positions.items()
        if ident and len(indexes) > 1
    }
    last = {ident: indexes[-1] for ident, indexes in positions.items()}
    kept = [
        rec
        for i, rec in enumerate(records)
        if not rec.get(key) or last[str(rec[key])] == i
    ]
    return kept, dupes


def portable_path(raw: str, source_base: Path) -> str:
    path = Path(raw)
    if not path.is_absolute():
        path = source_base / path
    return Path(os.path.relpath(path.resolve(), OUT.resolve())).as_posix()


def rewrite_pair_paths(rec: Dict[str, Any], source_base: Path) -> Dict[str, Any]:
    out = dict(rec)
    for key in (
        "state_a",
        "state_b",
        "icm_a",
        "icm_b",
        "render_a",
        "render_b",
    ):
        if out.get(key):
            out[key] = portable_path(str(out[key]), source_base)
    return out


def git_metadata() -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    existing_meta_path = OUT / "manifest.json"
    existing_meta = (
        json.loads(existing_meta_path.read_text(encoding="utf-8"))
        if existing_meta_path.exists()
        else {}
    )

    # Remove the task-ID collision before freezing diagnostics.
    t1, t1_text, t1_start, t1_end = load_tasks_js(T1_TASKS)
    t1_clean, task_dupes = dedupe_keep_last(t1, "id")
    if task_dupes:
        write_tasks_js(T1_TASKS, t1_clean, t1_text, t1_start, t1_end)

    labels = read_jsonl(T1_LABELS)
    labels_clean, label_dupes = dedupe_keep_last(labels, "sample_id")
    if label_dupes:
        write_jsonl(T1_LABELS, labels_clean)
    t1_diagnostic = [
        {
            **rec,
            "state": portable_path(str(rec["state"]), T1_LABELS.parent),
        }
        for rec in labels_clean
    ]
    t1_diagnostic_path = OUT / "t1_diagnostic.jsonl"
    write_jsonl(t1_diagnostic_path, t1_diagnostic)

    pairs = read_jsonl(PAIR_MANIFEST)
    train = [r for r in pairs if r.get("split") == "train"]
    val = [r for r in pairs if r.get("split") == "val"]
    train_views = {str(r.get("view_key") or "") for r in train}
    val_views = {str(r.get("view_key") or "") for r in val}
    overlap = sorted(train_views & val_views)
    if overlap:
        raise SystemExit(f"train/val view_key overlap: {overlap}")

    perturb_val = [
        rewrite_pair_paths(rec, PAIR_MANIFEST.parent)
        for rec in val
    ]
    perturb_val_path = OUT / "perturb_val.jsonl"
    write_jsonl(perturb_val_path, perturb_val)

    t2, _, _, _ = load_tasks_js(T2_TASKS)
    colleague = []
    for task in t2:
        if task.get("source") != "colleague_pairs":
            continue
        prefer = str(task.get("gold_prefer") or "").lower()
        if prefer not in ("a", "b"):
            role_a = str(task.get("colleague_role_a") or "")
            role_b = str(task.get("colleague_role_b") or "")
            if role_a == "perfect":
                prefer = "a"
            elif role_b == "perfect":
                prefer = "b"
        if prefer not in ("a", "b"):
            raise SystemExit(f"missing colleague gold preference: {task.get('id')}")
        colleague.append(
            rewrite_pair_paths(
                {
                    "pair_id": task["id"],
                    "view_key": task.get("view_key", ""),
                    "split": "challenge",
                    "level": "colleague",
                    "state_a": task["state_a"],
                    "state_b": task["state_b"],
                    "label": {"prefer": prefer},
                    "source": "colleague_pairs",
                },
                T2_TASKS.parent,
            )
        )
    colleague_path = OUT / "colleague_challenge.jsonl"
    write_jsonl(colleague_path, colleague)

    pair_ids = [str(r.get("pair_id") or "") for r in colleague]
    if len(pair_ids) != len(set(pair_ids)):
        raise SystemExit("duplicate colleague pair_id")
    colleague_views = {str(r.get("view_key") or "") for r in colleague}
    colleague_train_overlap = sorted(colleague_views & train_views)
    if colleague_train_overlap:
        raise SystemExit(
            f"colleague challenge overlaps perturb train: {colleague_train_overlap}"
        )

    duplicate_details = (
        existing_meta.get("sources", {})
        .get("t1", {})
        .get("duplicate_ids_removed", {})
    )
    for ident, rows in task_dupes.items():
        duplicate_details[ident] = {
            "kept_state": rows[-1].get("state"),
            "dropped_states": [r.get("state") for r in rows[:-1]],
        }

    metadata = {
        "schema_version": 1,
        "kind": "weak_supervision_proxy",
        "warning": "Not human gold; do not report as test or shipping KPI.",
        "frozen_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "git": git_metadata(),
        "sources": {
            "pairs_perturb": {
                "path": str(PAIR_MANIFEST.relative_to(ROOT).as_posix()),
                "sha256": sha256(PAIR_MANIFEST),
                "train_pairs": len(train),
                "val_pairs": len(val),
                "train_views": len(train_views),
                "val_views": len(val_views),
                "view_overlap": len(overlap),
                "level_counts_val": dict(
                    sorted(Counter(str(r.get("level") or "?") for r in val).items())
                ),
            },
            "t1": {
                "tasks": len(t1_clean),
                "labels": len(labels_clean),
                "duplicate_ids_removed": duplicate_details,
            },
            "colleague": {
                "pairs": len(colleague),
                "views": len(colleague_views),
                "perturb_train_view_overlap": len(colleague_train_overlap),
            },
        },
        "artifacts": {
            "perturb_val": {
                "path": str(perturb_val_path.relative_to(ROOT).as_posix()),
                "sha256": sha256(perturb_val_path),
            },
            "colleague_challenge": {
                "path": str(colleague_path.relative_to(ROOT).as_posix()),
                "sha256": sha256(colleague_path),
            },
            "t1_diagnostic": {
                "path": str(t1_diagnostic_path.relative_to(ROOT).as_posix()),
                "sha256": sha256(t1_diagnostic_path),
            },
        },
    }
    meta_path = OUT / "manifest.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"T1 tasks: {len(t1)} -> {len(t1_clean)}")
    print(f"T1 labels: {len(labels)} -> {len(labels_clean)}")
    print(f"perturb train/val: {len(train)}/{len(val)}, overlap=0")
    print(f"colleague challenge: {len(colleague)}, train overlap=0")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
