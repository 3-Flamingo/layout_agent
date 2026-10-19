"""Freeze and validate human-gold annotation subsets.

The annotation packages are intentionally larger than the shipping gate.  This
module selects a deterministic, view-isolated gold subset without rewriting the
packages colleagues are actively using.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


T1_TARGET = 15
T2_TARGET = 80


def load_tasks_js(path: Path) -> List[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    marker = "window.TASKS = "
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"{path}: missing {marker!r}")
    tasks, _ = json.JSONDecoder().raw_decode(text[start + len(marker) :])
    if not isinstance(tasks, list):
        raise ValueError(f"{path}: window.TASKS must be an array")
    return [dict(item) for item in tasks]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = [json.dumps(dict(row), ensure_ascii=False) for row in rows]
    path.write_text("\n".join(encoded) + ("\n" if encoded else ""), encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _unique_by_id(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    dupes: List[str] = []
    for task in tasks:
        ident = str(task.get("id") or "")
        if not ident:
            raise ValueError("annotation task without id")
        if ident in out:
            dupes.append(ident)
        out[ident] = dict(task)
    if dupes:
        raise ValueError(f"duplicate annotation task ids: {sorted(set(dupes))}")
    return out


def _decorate(task: Mapping[str, Any], source: str) -> Dict[str, Any]:
    out = dict(task)
    out["gold_split"] = "test"
    out["gold_source"] = source
    return out


def write_js_array(path: Path, name: str, values: Sequence[Any]) -> None:
    Path(path).write_text(
        f"window.{name} = {json.dumps(list(values), ensure_ascii=False)};\n",
        encoding="utf-8",
    )


def _unique_pair_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    identity_fields = ("view_key", "state_a", "state_b", "render_a", "render_b")
    for raw in rows:
        row = dict(raw)
        ident = str(row.get("pair_id") or row.get("id") or "")
        if not ident:
            raise ValueError("pair row without pair_id")
        if ident in unique:
            before = tuple(unique[ident].get(key) for key in identity_fields)
            after = tuple(row.get(key) for key in identity_fields)
            if before != after:
                raise ValueError(f"conflicting duplicate pair_id: {ident}")
            continue
        unique[ident] = row
    return list(unique.values())


def pair_row_to_t2_task(
    row: Mapping[str, Any],
    review_task: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a T2 gold task from a reviewed human_pairs row."""
    review = dict(review_task or {})
    ident = str(row.get("pair_id") or review.get("id") or "")
    if not ident:
        raise ValueError("human_pairs row without pair_id")
    task = {
        "id": ident,
        "type": "prefer",
        "view_key": str(row.get("view_key") or review.get("view_key") or ""),
        "render_a": str(review.get("render_a") or f"renders/{ident}_a.png"),
        "render_b": str(review.get("render_b") or f"renders/{ident}_b.png"),
        "state_a": str(row.get("state_a") or review.get("state_a") or ""),
        "state_b": str(row.get("state_b") or review.get("state_b") or ""),
        "drawing_url": str(review.get("drawing_url") or row.get("drawing_url") or ""),
        "drawing_url_alt": str(review.get("drawing_url_alt") or ""),
        "caption_a": str(review.get("caption_a") or "左 · 缺陷"),
        "caption_b": str(review.get("caption_b") or "右 · 完美"),
        "source": str(row.get("source") or review.get("source") or ""),
        "split": "test",
        "pose_delta": int(row.get("pose_delta") or review.get("pose_delta") or 0),
        "n_ann": int(row.get("n_ann") or review.get("n_ann") or 0),
        "quality": int(row.get("quality") or 0),
        "solution_id": str(row.get("solution_id") or review.get("solution_id") or ""),
        "solution_name": str(row.get("solution_name") or review.get("solution_name") or ""),
        "view_type": str(row.get("view_type") or review.get("view_type") or ""),
        "view_uuid": str(review.get("view_uuid") or ""),
        "tenant_id": str(row.get("tenant_id") or review.get("tenant_id") or ""),
        "part_name": str(review.get("part_name") or ""),
        "tree_uuid": str(review.get("tree_uuid") or ""),
        "perturbed_eid": row.get("perturbed_eid", review.get("perturbed_eid")),
        "perfect_eid": row.get("perfect_eid", review.get("perfect_eid")),
        "sides_swapped": bool(review.get("sides_swapped")),
        "annotation_note": "human_pairs test; left=defect, right=perfect",
    }
    return _decorate(task, "human_pairs_test")


def select_t2_gold(
    pairs: Sequence[Mapping[str, Any]],
    train_view_keys: Sequence[str],
    *,
    target: int = T2_TARGET,
    seed: int = 17,
    discarded_ids: Iterable[str] | None = None,
    max_per_solution: int = 4,
    review_tasks: Sequence[Mapping[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Select T2 gold from reviewed human_pairs test rows.

    Prefer larger left/right pose_delta (real defect vs perfect) and cap how
    many pairs come from one solution so the set is not a few crowded drawings.
    Perturb-val and the retired OSS supplement are not eligible.
    """
    train_views = set(train_view_keys)
    discarded = {str(item) for item in (discarded_ids or []) if item}
    review_by_id = {
        str(task.get("id") or ""): dict(task)
        for task in (review_tasks or [])
        if task.get("id")
    }
    eligible: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in _unique_pair_rows(pairs):
        ident = str(raw.get("pair_id") or "")
        if not ident or ident in discarded or ident in seen_ids:
            continue
        if str(raw.get("split") or "test") != "test":
            continue
        if str(raw.get("review_flag") or "ok") not in {"ok", ""}:
            continue
        view_key = str(raw.get("view_key") or "")
        if not view_key:
            continue
        if view_key in train_views:
            raise ValueError(f"T2 gold candidate leaks into train: {view_key}")
        prefer = str((raw.get("label") or {}).get("prefer") or "").lower()
        if prefer not in {"a", "b", "tie"}:
            continue
        seen_ids.add(ident)
        eligible.append(raw)
    eligible.sort(
        key=lambda row: (
            -int(row.get("pose_delta") or 0),
            -int(row.get("n_ann") or 0),
            -int(row.get("quality") or 0),
            _rank(str(row.get("pair_id") or ""), seed),
        )
    )

    cap = max(1, int(max_per_solution))
    selected: List[Dict[str, Any]] = []
    while cap <= max(target, int(max_per_solution)):
        selected = []
        used_views: set[str] = set()
        per_solution: Counter[str] = Counter()
        for row in eligible:
            view_key = str(row.get("view_key") or "")
            if view_key in used_views:
                continue
            solution = str(row.get("solution_id") or "")
            if solution and per_solution[solution] >= cap:
                continue
            selected.append(pair_row_to_t2_task(row, review_by_id.get(str(row.get("pair_id")))))
            used_views.add(view_key)
            if solution:
                per_solution[solution] += 1
            if len(selected) >= target:
                return selected
        cap += 1
    raise ValueError(
        f"only {len(selected)} reviewed human_pairs test tasks available; need {target}."
    )


def select_t1_gold(
    tasks: Sequence[Mapping[str, Any]],
    holdout_view_keys: Sequence[str],
    train_view_keys: Sequence[str],
    *,
    target: int | None = T1_TARGET,
    seed: int = 17,
    discarded_ids: Iterable[str] | None = None,
    grades: Mapping[str, int] | None = None,
    blocked_view_keys: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    """Select T1 gold.

    The shipping minimum is 15 distinct eval-only views. When human grades are
    supplied, ``target=None`` keeps every reviewed non-discard item (Spearman
    on n=15 is too noisy). Perturb-train views may be promoted into gold; they
    are then forbidden for later training. Human-pairs train/test views stay
    out via ``blocked_view_keys``.
    """
    task_by_id = _unique_by_id(tasks)
    holdout = set(holdout_view_keys)
    train_views = set(train_view_keys)
    blocked = {str(item) for item in (blocked_view_keys or []) if item}
    discarded = {str(item) for item in (discarded_ids or []) if item}
    by_view: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for task in task_by_id.values():
        ident = str(task.get("id") or "")
        if ident in discarded:
            continue
        if grades is not None and ident not in grades:
            continue
        if (
            task.get("source") == "colleague_pairs"
            or ident.startswith("colleague_")
        ):
            continue
        view_key = str(task.get("view_key") or "")
        if not view_key or view_key in blocked:
            continue
        if grades is None and view_key in train_views:
            continue
        by_view[view_key].append(task)

    selected: List[Dict[str, Any]] = []
    used_views: set[str] = set()
    need = T1_TARGET if target is None else int(target)

    def _source_for(view_key: str) -> str:
        if view_key in holdout:
            return "perturb_holdout"
        if view_key in train_views:
            return "reviewed_perturb_train"
        return "non_train_view"

    def pick_hash(view_keys: Iterable[str], source: str) -> None:
        limit = need if target is not None else 10**9
        for view_key in sorted(view_keys, key=lambda key: _rank(key, seed)):
            if len(selected) >= limit or view_key in used_views:
                continue
            choices = sorted(
                by_view.get(view_key, []),
                key=lambda task: _rank(str(task["id"]), seed),
            )
            if not choices:
                continue
            selected.append(_decorate(choices[0], source))
            used_views.add(view_key)

    if grades is None:
        pick_hash(holdout, "perturb_holdout")
        pick_hash(set(by_view), "non_train_view")
        if len(selected) < need:
            raise ValueError(
                f"only {len(selected)} leak-free T1 views available; need {need}"
            )
        return selected

    if target is None:
        items: List[Dict[str, Any]] = []
        for view_key, choices in by_view.items():
            source = _source_for(view_key)
            for task in sorted(choices, key=lambda row: _rank(str(row["id"]), seed)):
                items.append(_decorate(task, source))
        items.sort(key=lambda row: _rank(str(row["id"]), seed))
        if not items:
            raise ValueError("no reviewed T1 tasks available")
        return items

    view_choice: Dict[str, Dict[str, Any]] = {}
    for view_key, choices in by_view.items():
        ranked = sorted(choices, key=lambda task: _rank(str(task["id"]), seed))
        view_choice[view_key] = ranked[0]

    def consume(view_keys: Sequence[str]) -> None:
        buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for view_key in view_keys:
            task = view_choice.get(view_key)
            if task is None or view_key in used_views:
                continue
            grade = int(grades[str(task["id"])])
            buckets[grade].append(task)
        for grade in buckets:
            buckets[grade].sort(
                key=lambda task: (
                    0 if str(task.get("view_key") or "") in holdout else 1,
                    _rank(str(task["id"]), seed),
                )
            )
        progressed = True
        while len(selected) < need and progressed:
            progressed = False
            for grade in (1, 2, 3, 4, 5):
                if len(selected) >= need:
                    break
                pool = buckets.get(grade) or []
                while pool:
                    task = pool.pop(0)
                    view_key = str(task.get("view_key") or "")
                    if view_key in used_views:
                        continue
                    selected.append(_decorate(task, _source_for(view_key)))
                    used_views.add(view_key)
                    progressed = True
                    break

    consume(sorted(holdout))
    if len(selected) < need:
        consume(sorted(view_choice))
    if len(selected) < need:
        raise ValueError(
            f"only {len(selected)} reviewed leak-free T1 views available; need {need}"
        )
    return selected


@dataclass(frozen=True)
class LabelValidation:
    frozen: List[Dict[str, Any]]
    missing_ids: List[str]
    conflicting_ids: List[str]
    unknown_ids: List[str]
    discarded_ids: List[str]
    agreement: Dict[str, Any]

    @property
    def ready(self) -> bool:
        return not self.missing_ids and not self.conflicting_ids


def _label_key(kind: str, row: Mapping[str, Any]) -> str:
    return str(row.get("pair_id" if kind == "t2" else "sample_id") or "")


def _is_discard(label: Mapping[str, Any]) -> bool:
    if label.get("discard") is True:
        return True
    return str(label.get("prefer") or "").lower() in {"drop", "delete", "discard"}


def _label_value(kind: str, row: Mapping[str, Any]) -> Any:
    label = row.get("label") if isinstance(row.get("label"), dict) else {}
    if kind == "t2":
        value = str(label.get("prefer") or "").lower()
        if value not in {"a", "b", "tie"}:
            raise ValueError(f"invalid T2 prefer={value!r}")
        return value
    try:
        value = int(label.get("grade"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid T1 grade={label.get('grade')!r}") from exc
    if value < 1 or value > 5:
        raise ValueError(f"invalid T1 grade={value}; expected 1..5")
    return value


def _same_path(left: object, right: object) -> bool:
    if not left or not right:
        return False
    return os.path.normcase(os.path.abspath(str(left))) == os.path.normcase(
        os.path.abspath(str(right))
    )


def _validate_label_identity(
    kind: str,
    ident: str,
    task: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    fields = ("state_a", "state_b") if kind == "t2" else ("state",)
    for field in fields:
        expected = task.get(field)
        actual = row.get(field)
        if expected and not _same_path(expected, actual):
            raise ValueError(
                f"{kind} {ident}: exported {field} does not match frozen task"
            )
    if kind == "t2" and row.get("view_key") not in (None, "", task.get("view_key")):
        raise ValueError(f"{kind} {ident}: exported view_key does not match frozen task")
    for field in fields:
        expected_hash = str(task.get(f"{field}_sha256") or "")
        expected_path = task.get(field)
        if expected_hash and expected_path:
            path = Path(str(expected_path))
            if not path.is_file() or file_sha256(path) != expected_hash:
                raise ValueError(f"{kind} {ident}: frozen {field} artifact changed")


def validate_labels(
    kind: str,
    selected_tasks: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> LabelValidation:
    """Filter full-package exports to gold and reject unresolved disagreement."""
    if kind not in {"t1", "t2"}:
        raise ValueError("kind must be t1 or t2")
    selected = {
        str(task.get("id") or ""): dict(task)
        for task in selected_tasks
    }
    observations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    unknown: set[str] = set()
    discarded: set[str] = set()
    for raw in label_rows:
        row = dict(raw)
        ident = _label_key(kind, row)
        if not ident:
            raise ValueError(f"{kind} label without id")
        if ident not in selected:
            unknown.add(ident)
            continue
        label = row.get("label") if isinstance(row.get("label"), dict) else {}
        annotator = str(label.get("annotator") or "").strip()
        if not annotator or annotator.lower() == "unknown":
            raise ValueError(f"{kind} {ident}: annotator is required")
        if _is_discard(label):
            discarded.add(ident)
            continue
        _validate_label_identity(kind, ident, selected[ident], row)
        value = _label_value(kind, row)
        observations[ident].append({"row": row, "value": value, "annotator": annotator})

    frozen: List[Dict[str, Any]] = []
    conflicts: List[str] = []
    overlaps = agreements = 0
    t1_abs_diff: List[float] = []
    for ident in selected:
        obs = observations.get(ident, [])
        values = [item["value"] for item in obs]
        if len(values) >= 2:
            overlaps += 1
            if len(set(values)) == 1:
                agreements += 1
            elif kind == "t1":
                t1_abs_diff.append(float(max(values) - min(values)))
        if not obs:
            continue
        if len(set(values)) != 1:
            conflicts.append(ident)
            continue
        task = selected[ident]
        canonical = {
            ("pair_id" if kind == "t2" else "sample_id"): ident,
            "label": dict(obs[0]["row"].get("label") or {}),
            "gold_source": task.get("gold_source"),
            "annotation_count": len(obs),
        }
        if kind == "t2":
            canonical.update(
                {
                    "view_key": task.get("view_key", ""),
                    "state_a": task.get("state_a"),
                    "state_b": task.get("state_b"),
                }
            )
        else:
            canonical["state"] = task.get("state")
        frozen.append(canonical)

    missing = sorted(set(selected) - set(observations))
    agreement: Dict[str, Any] = {
        "overlap_items": overlaps,
        "exact_agreements": agreements,
        "exact_agreement_rate": agreements / overlaps if overlaps else None,
    }
    if kind == "t1":
        agreement["mean_conflict_range"] = (
            sum(t1_abs_diff) / len(t1_abs_diff) if t1_abs_diff else 0.0
        )
    return LabelValidation(
        frozen=sorted(frozen, key=lambda row: _label_key(kind, row)),
        missing_ids=missing,
        conflicting_ids=sorted(conflicts),
        unknown_ids=sorted(unknown),
        discarded_ids=sorted(discarded),
        agreement=agreement,
    )


def selection_summary(
    t1: Sequence[Mapping[str, Any]],
    t2: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "t1": {
            "tasks": len(t1),
            "views": len({str(row.get("view_key") or "") for row in t1}),
            "sources": dict(Counter(str(row.get("gold_source") or "") for row in t1)),
        },
        "t2": {
            "tasks": len(t2),
            "views": len({str(row.get("view_key") or "") for row in t2}),
            "sources": dict(Counter(str(row.get("gold_source") or "") for row in t2)),
        },
    }
