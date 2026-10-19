#!/usr/bin/env python3
"""Freeze and validate the 80-T2 / 15-T1 human-gold protocol.

This script never rewrites annotate_t1/tasks.js or annotate_t2/tasks.js, so it
is safe to run while colleagues are labeling the larger packages.

Examples:
  python scripts/evalset_human_gold.py freeze
  python scripts/evalset_human_gold.py validate \
      --t1-labels exports/t1_a.jsonl exports/t1_b.jsonl \
      --t2-labels exports/t2_a.jsonl exports/t2_b.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.evalset.human_gold import (  # noqa: E402
    T1_TARGET,
    T2_TARGET,
    file_sha256,
    load_tasks_js,
    read_jsonl,
    select_t1_gold,
    select_t2_gold,
    selection_summary,
    validate_labels,
    write_js_array,
    write_jsonl,
    _is_discard,
)
from layout_agent.evalset.isolation import (  # noqa: E402
    colleague_challenge_path,
    colleague_challenge_views,
    forbidden_training_views,
    human_pairs_test_views,
    human_pairs_train_views,
    write_forbidden_train_views,
)

EVALSYS = ROOT / "data" / "evalsys"
GOLD = EVALSYS / "human_gold"
EXPORTS = GOLD / "exports"
PAIR_MANIFEST = EVALSYS / "pairs_perturb" / "manifest.jsonl"
HUMAN_PAIRS_TEST = EVALSYS / "human_pairs" / "test.jsonl"
REVIEW = EVALSYS / "review_human_pairs"
T1_TASKS = EVALSYS / "annotate_t1" / "tasks.js"
T2_TASKS = EVALSYS / "annotate_t2" / "tasks.js"
T2_SUPPLEMENT = EVALSYS / "annotate_t2_gold_supplement"
HOLDOUT = EVALSYS / "holdout_views.txt"
TEST_META = EVALSYS / "test_manifest.json"
T1_ANNOTATOR = "on_device_reviewer"
T2_ANNOTATOR = "human_pairs_reviewer"


def _train_views(rows: Iterable[Dict[str, Any]]) -> List[str]:
    return sorted(
        {
            str(row.get("view_key") or "")
            for row in rows
            if row.get("split") == "train" and row.get("view_key")
        }
    )


def _retire_t2_supplement() -> None:
    """Drop the unqualified OSS/synth supplement from the gold protocol."""
    T2_SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "tools" / "annotate.html", T2_SUPPLEMENT / "annotate.html")
    (T2_SUPPLEMENT / "tasks.js").write_text(
        'window.DATASET_ID = "annotate_t2_gold_supplement";\n'
        "window.TASKS = [];\n"
        "window.SUPPLEMENT_RETIRED = true;\n",
        encoding="utf-8",
    )
    write_js_array(T2_SUPPLEMENT / "gold_ids.js", "GOLD_IDS", [])
    (T2_SUPPLEMENT / "discards.js").write_text(
        "window.DISCARDS = {};\n",
        encoding="utf-8",
    )


def _stamp_annotator(label: Dict[str, Any], default: str) -> Dict[str, Any]:
    out = dict(label)
    name = str(out.get("annotator") or "").strip()
    if not name or name.lower() == "unknown":
        out["annotator"] = default
    return out


def _browser_label_maps() -> Dict[str, Any]:
    dump = GOLD / "_browser_labels.json"
    if dump.is_file():
        payload = json.loads(dump.read_text(encoding="utf-8"))
        return dict(payload.get("values") or {})
    return {}


def export_package_labels() -> Dict[str, Path]:
    """Write Chrome localStorage labels to jsonl with a non-unknown annotator."""
    EXPORTS.mkdir(parents=True, exist_ok=True)
    t1_path = EXPORTS / "labels_annotate_t1.jsonl"
    t2_path = EXPORTS / "labels_annotate_t2.jsonl"
    sup_path = EXPORTS / "labels_annotate_t2_gold_supplement.jsonl"
    values = _browser_label_maps()
    if not values:
        if not t1_path.is_file():
            raise FileNotFoundError(
                "need data/evalsys/human_gold/_browser_labels.json or "
                "exports/labels_annotate_t1.jsonl"
            )
        return {"t1": t1_path, "t2": t2_path, "t2_supplement": sup_path}
    packages = {
        "t1": (
            load_tasks_js(T1_TASKS),
            values.get("labels_annotate_t1") or {},
            T1_ANNOTATOR,
            "sample_id",
        ),
        "t2": (
            load_tasks_js(T2_TASKS),
            values.get("labels_annotate_t2") or {},
            T1_ANNOTATOR,
            "pair_id",
        ),
        "t2_supplement": (
            load_tasks_js(T2_SUPPLEMENT / "tasks.js")
            if (T2_SUPPLEMENT / "tasks.js").is_file()
            else [],
            values.get("labels_annotate_t2_gold_supplement") or {},
            T1_ANNOTATOR,
            "pair_id",
        ),
    }
    written: Dict[str, Path] = {}
    for kind, (tasks, labels, annotator, id_field) in packages.items():
        by_id = {str(task.get("id") or ""): task for task in tasks}
        rows: List[Dict[str, Any]] = []
        for ident, raw in labels.items():
            if not isinstance(raw, dict):
                continue
            label = _stamp_annotator(dict(raw), annotator)
            row: Dict[str, Any] = {id_field: ident, "label": label}
            task = by_id.get(str(ident))
            if task:
                if id_field == "sample_id":
                    row["state"] = task.get("state")
                else:
                    row["view_key"] = task.get("view_key", "")
                    row["state_a"] = task.get("state_a")
                    row["state_b"] = task.get("state_b")
            rows.append(row)
        path = EXPORTS / f"labels_annotate_{kind if kind != 't2_supplement' else 't2_gold_supplement'}.jsonl"
        if kind == "t1":
            path = EXPORTS / "labels_annotate_t1.jsonl"
        elif kind == "t2":
            path = EXPORTS / "labels_annotate_t2.jsonl"
        write_jsonl(path, rows)
        written[kind] = path
    return written


def _t1_review_signals(label_rows: Sequence[Dict[str, Any]]) -> tuple:
    discarded: List[str] = []
    grades: Dict[str, int] = {}
    for row in label_rows:
        ident = str(row.get("sample_id") or "")
        label = row.get("label") if isinstance(row.get("label"), dict) else {}
        if not ident:
            continue
        if _is_discard(label):
            discarded.append(ident)
            continue
        try:
            grade = int(label.get("grade"))
        except (TypeError, ValueError):
            continue
        if 1 <= grade <= 5:
            grades[ident] = grade
    return discarded, grades


def _absolutize_states(kind: str, tasks: List[Dict[str, Any]]) -> None:
    fields = ("state_a", "state_b") if kind == "t2" else ("state",)
    for task in tasks:
        for field in fields:
            raw = Path(str(task.get(field) or ""))
            if not str(raw):
                continue
            if not raw.is_absolute():
                raw = (ROOT / raw).resolve()
            task[field] = str(raw.as_posix())


def _attach_artifact_hashes(
    kind: str,
    tasks: List[Dict[str, Any]],
) -> None:
    for task in tasks:
        state_fields = ("state_a", "state_b") if kind == "t2" else ("state",)
        for field in state_fields:
            path = Path(str(task.get(field) or ""))
            if not path.is_file():
                raise FileNotFoundError(f"{task.get('id')}: missing {field} {path}")
            task[f"{field}_sha256"] = file_sha256(path)
        render_fields = ("render_a", "render_b") if kind == "t2" else ("render",)
        if task.get("gold_source") == "human_pairs_test":
            package = REVIEW
        elif task.get("source") == "perturb_val_supplement":
            package = T2_SUPPLEMENT
        else:
            package = EVALSYS / ("annotate_t2" if kind == "t2" else "annotate_t1")
        task["render_package"] = str(package.relative_to(ROOT).as_posix())
        for field in render_fields:
            path = package / str(task.get(field) or "")
            if not path.is_file():
                raise FileNotFoundError(f"{task.get('id')}: missing {field} {path}")
            task[f"{field}_sha256"] = file_sha256(path)


def _verify_frozen_artifacts(kind: str, tasks: List[Dict[str, Any]]) -> None:
    for task in tasks:
        fields = (
            ("state_a", "state_b", "render_a", "render_b")
            if kind == "t2"
            else ("state", "render")
        )
        package = ROOT / str(task.get("render_package") or "")
        for field in fields:
            raw = str(task.get(field) or "")
            path = Path(raw) if field.startswith("state") else package / raw
            expected = str(task.get(f"{field}_sha256") or "")
            if not expected or not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"{kind} {task.get('id')}: frozen {field} changed")


def _write_meta(t1: List[Dict[str, Any]], t2: List[Dict[str, Any]]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    summary = selection_summary(t1, t2)
    train_views = set(_train_views(read_jsonl(PAIR_MANIFEST)))
    t1_views = {str(row.get("view_key") or "") for row in t1}
    t2_views = {str(row.get("view_key") or "") for row in t2}
    gold_views = t1_views | t2_views
    t2_train = sorted(train_views & t2_views)
    if t2_train:
        raise RuntimeError(f"T2 gold/train view leakage: {t2_train}")
    overlap = sorted(train_views & t1_views)
    hp_train = human_pairs_train_views(ROOT)
    hp_train_overlap = sorted(gold_views & hp_train)
    if hp_train_overlap:
        raise RuntimeError(f"gold/human_pairs train view leakage: {hp_train_overlap}")
    colleague_views = colleague_challenge_views(ROOT)
    colleague_overlap = sorted(gold_views & colleague_views)
    if colleague_overlap:
        raise RuntimeError(f"gold/colleague challenge leakage: {colleague_overlap}")
    forbidden = forbidden_training_views(ROOT)
    forbidden_path = write_forbidden_train_views(ROOT, forbidden)
    meta = {
        "schema_version": 1,
        "frozen_at": now,
        "status": "tasks_frozen; awaiting human labels",
        "protocol": {
            "t1": (
                "human 1-5 grade; all reviewed non-discard annotate_t1 items "
                "(minimum 15); extra views from perturb train are promoted into forbidden"
            ),
            "t2": (
                "human pairwise preference on frozen human_pairs test; "
                "80 high pose_delta eval-only pairs (left=defect, right=perfect)"
            ),
            "selection_seed": 17,
            "view_isolation": (
                "gold view_keys excluded from later training; T2 gold is a subset of "
                "human_pairs test; T1 gold may promote reviewed perturb-train views"
            ),
            "challenge_policy": (
                "colleague challenge dump is excluded from T1 gold and never selects "
                "checkpoints; T2 gold uses reviewed human_pairs test, not perturb/supplement"
            ),
            "adjudication": "duplicate annotations must agree; conflicts require adjudication",
            "label_validation": (
                "filter package exports to frozen gold ids; require annotator; "
                "duplicates must agree or be adjudicated; exported states must match frozen hashes"
            ),
            "annotator_policy": (
                "annotation UI name was blank; T1 stamped on_device_reviewer from "
                "Chrome labels; T2 stamped human_pairs_reviewer from reviewed test.jsonl"
            ),
            "t2_pool": "data/evalsys/review_human_pairs (human_pairs/test.jsonl)",
            "t2_supplement": "retired; all 12 items discarded as unqualified",
        },
        "tasks": {
            "t1": "data/evalsys/human_gold/t1_tasks.jsonl",
            "t2": "data/evalsys/human_gold/t2_tasks.jsonl",
        },
        "labels": {"t1": None, "t2": None},
        "summary": summary,
        "source_artifacts": {
            "annotate_t1_tasks": {
                "path": "data/evalsys/annotate_t1/tasks.js",
                "sha256": file_sha256(T1_TASKS),
            },
            "annotate_t2_tasks": {
                "path": "data/evalsys/annotate_t2/tasks.js",
                "sha256": file_sha256(T2_TASKS),
            },
            "pairs_manifest": {
                "path": "data/evalsys/pairs_perturb/manifest.jsonl",
                "sha256": file_sha256(PAIR_MANIFEST),
            },
            "human_pairs_test": {
                "path": "data/evalsys/human_pairs/test.jsonl",
                "sha256": file_sha256(HUMAN_PAIRS_TEST),
            },
            "review_human_pairs_tasks": {
                "path": "data/evalsys/review_human_pairs/tasks.js",
                "sha256": file_sha256(REVIEW / "tasks.js"),
            },
            "t2_supplement_tasks": {
                "path": "data/evalsys/annotate_t2_gold_supplement/tasks.js",
                "sha256": file_sha256(T2_SUPPLEMENT / "tasks.js"),
            },
            "frozen_t1_tasks": {
                "path": "data/evalsys/human_gold/t1_tasks.jsonl",
                "sha256": file_sha256(GOLD / "t1_tasks.jsonl"),
            },
            "frozen_t2_tasks": {
                "path": "data/evalsys/human_gold/t2_tasks.jsonl",
                "sha256": file_sha256(GOLD / "t2_tasks.jsonl"),
            },
            "colleague_challenge": {
                "path": "data/evalsys/proxy/colleague_challenge.jsonl",
                "sha256": (
                    file_sha256(colleague_challenge_path(ROOT))
                    if colleague_challenge_path(ROOT).is_file()
                    else None
                ),
            },
            "forbidden_train_views": {
                "path": "data/evalsys/forbidden_train_views.txt",
                "sha256": file_sha256(forbidden_path),
            },
        },
        "train_view_overlap": overlap,
        "t1_promoted_from_perturb_train": overlap,
        "human_pairs_train_overlap": hp_train_overlap,
        "colleague_challenge_overlap": colleague_overlap,
        "forbidden_view_count": len(forbidden),
        "colleague_challenge_views": len(colleague_views),
    }
    GOLD.mkdir(parents=True, exist_ok=True)
    (GOLD / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # test_manifest is the public pointer to the current protocol. Preserve the
    # old generation details under legacy_proxy_holdout for traceability.
    current = json.loads(TEST_META.read_text(encoding="utf-8")) if TEST_META.exists() else {}
    legacy = current.get("legacy_proxy_holdout", current)
    public_meta = {
        "frozen_at": now,
        "protocol": meta["protocol"],
        "n_t1_tasks": len(t1),
        "n_t2_pairs": len(t2),
        "gold_manifest": "data/evalsys/human_gold/manifest.json",
        "packages": {
            "t1": "data/evalsys/annotate_t1/",
            "t2": "data/evalsys/review_human_pairs/",
            "t2_perturb_package": "data/evalsys/annotate_t2/",
            "t2_supplement": "data/evalsys/annotate_t2_gold_supplement/",
        },
        "labels": {"t1": None, "t2": None},
        "status": meta["status"],
        "forbidden_train_views": "data/evalsys/forbidden_train_views.txt",
        "n_forbidden_views": len(forbidden),
        "legacy_proxy_holdout": legacy,
    }
    TEST_META.write_text(
        json.dumps(public_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta


def _write_t2_gold_labels(tasks: List[Dict[str, Any]]) -> Path:
    by_id = {str(row.get("pair_id") or ""): row for row in read_jsonl(HUMAN_PAIRS_TEST)}
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        ident = str(task.get("id") or "")
        src = by_id.get(ident)
        if src is None:
            raise ValueError(f"T2 gold {ident} is not in human_pairs/test.jsonl")
        prefer = str((src.get("label") or {}).get("prefer") or "").lower()
        if prefer not in {"a", "b", "tie"}:
            raise ValueError(f"T2 gold {ident}: invalid prefer={prefer!r}")
        rows.append(
            {
                "pair_id": ident,
                "view_key": task.get("view_key", ""),
                "state_a": task.get("state_a"),
                "state_b": task.get("state_b"),
                "label": {
                    "prefer": prefer,
                    "annotator": T2_ANNOTATOR,
                    "comment": "human_pairs test after review.html; left=defect, right=perfect",
                },
            }
        )
    path = EXPORTS / "labels_t2_gold_human_pairs.jsonl"
    EXPORTS.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)
    return path


def freeze() -> None:
    pair_rows = read_jsonl(PAIR_MANIFEST)
    train_views = _train_views(pair_rows)
    hp_train_views = sorted(human_pairs_train_views(ROOT))
    hp_test_views = sorted(human_pairs_test_views(ROOT))
    blocked_train = sorted(set(train_views) | set(hp_train_views))
    t1_blocked = sorted(set(hp_train_views) | set(hp_test_views))
    val_rows = [row for row in pair_rows if row.get("split") == "val"]
    val_views = sorted(
        {
            str(row.get("view_key") or "")
            for row in val_rows
            if row.get("view_key")
        }
    )
    HOLDOUT.write_text(
        "# Frozen human/perturb validation view_keys. Never train or select checkpoints on these.\n"
        + "\n".join(val_views)
        + "\n",
        encoding="utf-8",
    )
    export_package_labels()
    _retire_t2_supplement()
    t1_package = load_tasks_js(T1_TASKS)
    t1_rows = read_jsonl(EXPORTS / "labels_annotate_t1.jsonl")
    discarded, grades = _t1_review_signals(t1_rows)
    t1 = select_t1_gold(
        t1_package,
        val_views,
        train_views,
        target=None,
        seed=17,
        discarded_ids=discarded,
        grades=grades,
        blocked_view_keys=t1_blocked,
    )
    if len(t1) < T1_TARGET:
        raise RuntimeError(f"only {len(t1)} reviewed T1 gold tasks; need {T1_TARGET}")
    review_tasks = load_tasks_js(REVIEW / "tasks.js")
    t2 = select_t2_gold(
        read_jsonl(HUMAN_PAIRS_TEST),
        blocked_train,
        target=T2_TARGET,
        seed=17,
        review_tasks=review_tasks,
    )
    _absolutize_states("t1", t1)
    _absolutize_states("t2", t2)
    _attach_artifact_hashes("t1", t1)
    _attach_artifact_hashes("t2", t2)
    write_jsonl(GOLD / "t1_tasks.jsonl", t1)
    write_jsonl(GOLD / "t2_tasks.jsonl", t2)
    write_js_array(EVALSYS / "annotate_t1" / "gold_ids.js", "GOLD_IDS", [row["id"] for row in t1])
    write_js_array(EVALSYS / "annotate_t2" / "gold_ids.js", "GOLD_IDS", [])
    write_js_array(REVIEW / "gold_ids.js", "GOLD_IDS", [row["id"] for row in t2])
    _write_t2_gold_labels(t2)
    meta = _write_meta(t1, t2)
    print(json.dumps(meta["summary"], ensure_ascii=False, indent=2))
    print(f"frozen protocol -> {GOLD / 'manifest.json'}")


def _load_many(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in paths:
        rows.extend(read_jsonl(Path(raw)))
    return rows


def validate(t1_paths: List[str], t2_paths: List[str]) -> None:
    if not (GOLD / "manifest.json").exists():
        raise SystemExit("run `evalset_human_gold.py freeze` first")
    meta = json.loads((GOLD / "manifest.json").read_text(encoding="utf-8"))
    for artifact in (meta.get("source_artifacts") or {}).values():
        path = ROOT / str(artifact.get("path") or "")
        expected = artifact.get("sha256")
        if not expected:
            continue
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"frozen source artifact changed: {path}")
    t1_tasks = read_jsonl(GOLD / "t1_tasks.jsonl")
    t2_tasks = read_jsonl(GOLD / "t2_tasks.jsonl")
    _verify_frozen_artifacts("t1", t1_tasks)
    _verify_frozen_artifacts("t2", t2_tasks)
    report: Dict[str, Any] = {
        "validated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    }
    ready = True
    for kind, paths, tasks in (
        ("t1", t1_paths, t1_tasks),
        ("t2", t2_paths, t2_tasks),
    ):
        if not paths:
            report[kind] = {"ready": False, "reason": "no label files supplied"}
            ready = False
            continue
        result = validate_labels(kind, tasks, _load_many(paths))
        report[kind] = {
            "ready": result.ready,
            "frozen_labels": len(result.frozen),
            "missing_ids": result.missing_ids,
            "conflicting_ids": result.conflicting_ids,
            "discarded_ids": result.discarded_ids,
            "ignored_non_gold_ids": result.unknown_ids,
            "agreement": result.agreement,
        }
        if result.ready:
            write_jsonl(GOLD / f"{kind}_labels.jsonl", result.frozen)
        else:
            ready = False
    report["ready"] = ready
    (GOLD / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if ready:
        meta["status"] = "human_labels_frozen"
        meta["labels"] = {
            "t1": "data/evalsys/human_gold/t1_labels.jsonl",
            "t2": "data/evalsys/human_gold/t2_labels.jsonl",
        }
        (GOLD / "manifest.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        public = json.loads(TEST_META.read_text(encoding="utf-8"))
        public["status"] = "human_labels_frozen"
        public["labels"] = meta["labels"]
        TEST_META.write_text(
            json.dumps(public, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not ready:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    val = sub.add_parser("validate")
    val.add_argument("--t1-labels", nargs="*", default=[])
    val.add_argument("--t2-labels", nargs="*", default=[])
    args = parser.parse_args()
    if args.cmd == "freeze":
        freeze()
        validate(
            [str(EXPORTS / "labels_annotate_t1.jsonl")],
            [str(EXPORTS / "labels_t2_gold_human_pairs.jsonl")],
        )
    else:
        validate(args.t1_labels, args.t2_labels)


if __name__ == "__main__":
    main()
