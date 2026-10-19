#!/usr/bin/env python3
"""Fetch the colleague project's current layouts as a harder review-only batch.

The previous left/defective states were too poor, so formula ranking was easy.
Colleagues re-ran the layout algorithm; left vs perfect still differs, but the
gap is smaller. All drawings at the live project URL are treated as new data.

Snapshots go to a new dataset directory so existing train/test freeze files keep
pointing at the old snapshots. This batch is written to relayout.jsonl for the
review HTML only; it is not merged into train or test.

Previous 建议删除 view_keys are copied onto matching new pairs as review_flag=drop.
Old unlabeled pairs stay on the review page and are preset to 合理 in the UI.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evalset_ingest_human_pairs import (  # noqa: E402
    DRAWING_HOST,
    HUMAN_PAIRS,
    PROJECT_ID,
    RAW_ROOT,
    TENANT,
    _safe,
    build_index,
    build_sheet_pairs,
    convert_pair_views,
    fetch_solution_raw,
    list_project_solutions,
    pair_record,
    rel_to_root,
    review_view_keys,
)
from layout_agent.evalset.human_gold import read_jsonl, write_jsonl  # noqa: E402

RELAYOUT_DS = ROOT / "data" / "datasets" / "prod_colleague_pairs_enterprise_t3_relayout"
SOURCE = "colleague_enterprise_t3_relayout"
REGISTRY = HUMAN_PAIRS / "registry.json"


def previous_drop_view_keys() -> Set[str]:
    keys = set(review_view_keys("drop"))
    dropped_path = HUMAN_PAIRS / "dropped.jsonl"
    if dropped_path.exists():
        for row in read_jsonl(dropped_path):
            if str(row.get("review_flag") or "") == "drop":
                keys.add(str(row.get("view_key") or ""))
            if str(row.get("reason") or "") == "review_drop_low_quality":
                keys.add(str(row.get("view_key") or ""))
    keys.discard("")
    return keys


def ingest_relayout(reuse_raw: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    RELAYOUT_DS.mkdir(parents=True, exist_ok=True)
    (RELAYOUT_DS / "snapshots").mkdir(parents=True, exist_ok=True)
    solutions = list_project_solutions()
    catalog_path = RAW_ROOT / f"project_{PROJECT_ID}_solutions.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(solutions, ensure_ascii=False, indent=2), encoding="utf-8")
    drop_keys = previous_drop_view_keys()
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    ds_pairs: List[Dict[str, Any]] = []
    for row in solutions:
        sid = str(row.get("id") or "")
        name = str(row.get("name") or sid)
        if not row.get("solutionSolutionWebDocument"):
            print(f"SKIP {sid} {name}: no web document")
            dropped.append({"solution_id": sid, "name": name, "reason": "no_web_document"})
            continue
        print(f"FETCH {sid} {name}")
        try:
            meta, doc = fetch_solution_raw(sid, reuse_raw)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL fetch: {exc}")
            dropped.append({"solution_id": sid, "name": name, "reason": f"fetch:{exc}"})
            continue
        meta["_tenant_id"] = TENANT
        meta["_drawing_host"] = DRAWING_HOST
        by = build_index(doc)
        sheet_pairs = build_sheet_pairs(by)
        print(f"  cad drawings paired views: {len(sheet_pairs)}")
        for rec in sheet_pairs:
            item = convert_pair_views(rec, by, meta, RELAYOUT_DS)
            if item is None:
                dropped.append(
                    {
                        "solution_id": sid,
                        "uuid": rec.get("uuid"),
                        "reason": "convert_failed_or_unannotated",
                    }
                )
                continue
            vkey = f"{sid}/{rec['uuid']}"
            extra = {
                "drawing_url": f"{DRAWING_HOST}/?solutionId={sid}&tenantId={TENANT}",
                "solution_name": name,
                "perturbed_eid": item["perturbed_eid"],
                "perfect_eid": item["perfect_eid"],
            }
            if item["n_ann"] <= 0:
                dropped.append({"view_key": vkey, "reason": "no_annotations", **extra})
                continue
            if item["identical"] or item["pose_delta"] <= 0:
                dropped.append(
                    {
                        "view_key": vkey,
                        "reason": "identical_pose",
                        "pose_delta": item["pose_delta"],
                        **extra,
                    }
                )
                continue
            pair_id = f"relayout_{_safe(str(rec['uuid']))}_{sid[-6:]}_{rec['pair_index']}"
            record = pair_record(
                pair_id=pair_id,
                vkey=vkey,
                source=SOURCE,
                solution_id=sid,
                tenant_id=TENANT,
                dataset=str(RELAYOUT_DS.relative_to(ROOT).as_posix()),
                perturbed_state=rel_to_root(RELAYOUT_DS / item["perturbed_state"]),
                perfect_state=rel_to_root(RELAYOUT_DS / item["perfect_state"]),
                view_type=str(item.get("viewType") or ""),
                pose_delta_n=int(item["pose_delta"]),
                n_ann=int(item["n_ann"]),
                extra=extra,
            )
            record["split"] = "relayout"
            record["batch"] = "relayout"
            if vkey in drop_keys:
                record["review_flag"] = "drop"
                record["review_flag_reason"] = "previous_review_drop"
            ds_item = dict(item)
            ds_item["pair_id"] = pair_id
            ds_item["view_key"] = vkey
            ds_item["solution_id"] = sid
            if record.get("review_flag"):
                ds_item["review_flag"] = record["review_flag"]
            kept.append(record)
            ds_pairs.append(ds_item)
    (RELAYOUT_DS / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "prod_colleague_pairs_enterprise_t3_relayout",
                "project_id": PROJECT_ID,
                "tenant_id": TENANT,
                "n_pairs": len(ds_pairs),
                "pairs": ds_pairs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"relayout kept {len(kept)} dropped {len(dropped)} preset_drop {sum(1 for r in kept if r.get('review_flag')=='drop')}")
    return kept, dropped


def update_registry(kept: List[Dict[str, Any]], dropped: List[Dict[str, Any]]) -> None:
    if not REGISTRY.exists():
        return
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    data["relayout_review"] = {
        "fetched_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "dataset": str(RELAYOUT_DS.relative_to(ROOT).as_posix()),
        "source": SOURCE,
        "n_pairs": len(kept),
        "n_dropped_identical_or_unannotated": len(dropped),
        "n_preset_drop": sum(1 for row in kept if row.get("review_flag") == "drop"),
        "merged_into_train": False,
        "merged_into_test": False,
        "note": "Harder left/perfect pairs after layout-algorithm initial states. Review only.",
    }
    REGISTRY.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Reuse cached OSS documents (default: re-download live project docs).",
    )
    args = ap.parse_args()
    kept, dropped = ingest_relayout(reuse_raw=args.reuse_raw)
    write_jsonl(HUMAN_PAIRS / "relayout.jsonl", kept)
    write_jsonl(HUMAN_PAIRS / "relayout_dropped.jsonl", dropped)
    update_registry(kept, dropped)
    print(f"wrote {HUMAN_PAIRS / 'relayout.jsonl'} n={len(kept)}")


if __name__ == "__main__":
    main()
