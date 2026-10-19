"""Two human-pair train variants: real-only vs real + reviewed CMU synth.

Canonical files under data/evalsys/human_pairs/:

  train.jsonl              real colleague pairs only
  train_with_synth.jsonl   real pairs plus reviewed CMU perturbations
  cmu_perturb.jsonl        synth archive (review split); not a train loader
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from layout_agent.evalset.human_gold import read_jsonl, write_jsonl
from layout_agent.evalset.isolation import forbidden_training_views, jsonl_view_keys

TRAIN_REAL = "train.jsonl"
TRAIN_WITH_SYNTH = "train_with_synth.jsonl"
CMU_JSONL = "cmu_perturb.jsonl"
CMU_DENSE_JSONL = "cmu_perturb_dense.jsonl"


def human_pairs_dir(root: Path) -> Path:
    return Path(root) / "data" / "evalsys" / "human_pairs"


def prepare_synth_train_row(rec: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    out["split"] = "train"
    out["synthetic"] = True
    out["synth_kind"] = "cmu_perturb"
    out.pop("review_only", None)
    return out


def filter_synth_for_train(
    synth_rows: Sequence[Mapping[str, Any]],
    *,
    train_view_keys: Set[str],
    blocked_view_keys: Set[str],
    real_pair_ids: Set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept: List[Dict[str, Any]] = []
    n_blocked = n_outside = n_id = 0
    for rec in synth_rows:
        vkey = str(rec.get("view_key") or "")
        pid = str(rec.get("pair_id") or "")
        if not vkey or vkey in blocked_view_keys:
            n_blocked += 1
            continue
        if vkey not in train_view_keys:
            n_outside += 1
            continue
        if pid in real_pair_ids:
            n_id += 1
            continue
        kept.append(prepare_synth_train_row(rec))
    stats = {
        "synth_kept": len(kept),
        "synth_blocked": n_blocked,
        "synth_outside_train": n_outside,
        "synth_id_collision": n_id,
    }
    return kept, stats


def build_train_with_synth(
    real_rows: Sequence[Mapping[str, Any]],
    synth_rows: Sequence[Mapping[str, Any]],
    *,
    blocked_view_keys: Set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    train_keys = {str(r.get("view_key") or "") for r in real_rows if r.get("view_key")}
    real_ids = {str(r.get("pair_id") or "") for r in real_rows if r.get("pair_id")}
    synth_kept, stats = filter_synth_for_train(
        synth_rows,
        train_view_keys=train_keys,
        blocked_view_keys=blocked_view_keys,
        real_pair_ids=real_ids,
    )
    merged = [dict(r) for r in real_rows] + synth_kept
    stats["real"] = len(real_rows)
    stats["with_synth"] = len(merged)
    return merged, stats


def write_train_variants(root: Path, synth_filename: str = CMU_JSONL) -> Dict[str, Any]:
    """Rewrite train_with_synth.jsonl from current train.jsonl + a synth jsonl.

    Never mutates train.jsonl or test.jsonl.
    """
    hp = human_pairs_dir(root)
    real_path = hp / TRAIN_REAL
    if not real_path.exists():
        raise FileNotFoundError(real_path)
    real_rows = read_jsonl(real_path)
    cmu_path = hp / synth_filename
    synth_rows = read_jsonl(cmu_path) if cmu_path.exists() else []
    blocked = forbidden_training_views(root)
    merged, stats = build_train_with_synth(real_rows, synth_rows, blocked_view_keys=blocked)
    out_path = hp / TRAIN_WITH_SYNTH
    write_jsonl(out_path, merged)

    now = datetime.now(timezone.utc).astimezone().isoformat()
    registry_path = hp / "registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        counts = dict(registry.get("counts") or {})
        counts["train_pairs"] = len(real_rows)
        counts["train_pairs_real"] = len(real_rows)
        counts["train_pairs_with_synth"] = len(merged)
        counts["cmu_perturb"] = stats["synth_kept"]
        registry["counts"] = counts
        registry["train_variants"] = {
            "real": f"data/evalsys/human_pairs/{TRAIN_REAL}",
            "with_synth": f"data/evalsys/human_pairs/{TRAIN_WITH_SYNTH}",
            "synth_source": f"data/evalsys/human_pairs/{synth_filename}",
            "merged_at": now,
            "synth_blocked": stats["synth_blocked"],
            "synth_outside_train": stats["synth_outside_train"],
            "synth_id_collision": stats["synth_id_collision"],
        }
        registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if synth_filename == CMU_JSONL:
        meta_path = Path(root) / "data" / "evalsys" / "cmu_perturb" / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["review_only"] = False
            meta["not_in_train"] = False
            meta["merged_into"] = f"data/evalsys/human_pairs/{TRAIN_WITH_SYNTH}"
            meta["real_train"] = f"data/evalsys/human_pairs/{TRAIN_REAL}"
            meta["merged_at"] = now
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stats["test_overlap"] = len(jsonl_view_keys(hp / "test.jsonl") & {str(r.get("view_key") or "") for r in merged})
    stats["out"] = str(out_path.as_posix())
    return stats
