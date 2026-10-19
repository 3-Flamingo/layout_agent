#!/usr/bin/env python3
"""Apply discard marks from annotate.html exports.

Does not rewrite frozen tasks.js. Writes a sidecar discards.js so the
annotation page can hide the same items after a refresh, and records gold
hits under data/evalsys/human_gold/annotation_discards.jsonl.

  python scripts/evalset_apply_annotation_discards.py exports/labels_annotate_t2_*.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.evalset.human_gold import (  # noqa: E402
    _is_discard,
    load_tasks_js,
    read_jsonl,
    write_jsonl,
)

PACKAGES = {
    "annotate_t1": ROOT / "data" / "evalsys" / "annotate_t1",
    "annotate_t2": ROOT / "data" / "evalsys" / "annotate_t2",
    "annotate_t2_gold_supplement": ROOT / "data" / "evalsys" / "annotate_t2_gold_supplement",
}
GOLD_T1 = ROOT / "data" / "evalsys" / "human_gold" / "t1_tasks.jsonl"
GOLD_T2 = ROOT / "data" / "evalsys" / "human_gold" / "t2_tasks.jsonl"
OUT = ROOT / "data" / "evalsys" / "human_gold" / "annotation_discards.jsonl"


def _row_id(row: Dict[str, Any]) -> str:
    return str(row.get("pair_id") or row.get("sample_id") or "")


def _dataset_id(path: Path) -> str:
    name = path.name
    for key in sorted(PACKAGES, key=len, reverse=True):
        if key in name:
            return key
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exports", nargs="+", type=Path)
    args = parser.parse_args()
    gold_ids = {
        str(row.get("id") or "")
        for row in read_jsonl(GOLD_T1) + read_jsonl(GOLD_T2)
        if row.get("id")
    }
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    existing = read_jsonl(OUT) if OUT.exists() else []
    by_id = {_row_id(row): row for row in existing if _row_id(row)}
    report: Dict[str, Any] = {"applied_at": now, "packages": {}}
    for export in args.exports:
        rows = read_jsonl(export)
        drops = [
            row
            for row in rows
            if isinstance(row.get("label"), dict) and _is_discard(row["label"])
        ]
        dsid = _dataset_id(export)
        package = PACKAGES.get(dsid)
        gold_hits = []
        for row in drops:
            ident = _row_id(row)
            if not ident:
                continue
            rec = {
                "id": ident,
                "view_key": row.get("view_key", ""),
                "reason": row.get("label", {}).get("discard_reason")
                or row.get("label", {}).get("prefer")
                or "other",
                "annotator": row.get("label", {}).get("annotator", ""),
                "exported_from": str(export),
                "applied_at": now,
                "gold": ident in gold_ids,
            }
            by_id[ident] = rec
            if ident in gold_ids:
                gold_hits.append(ident)
        if package and package.is_dir():
            payload = {
                ident: {
                    "reason": rec["reason"],
                    "annotator": rec.get("annotator", ""),
                }
                for ident, rec in by_id.items()
                if ident
                in {str(task.get("id") or "") for task in load_tasks_js(package / "tasks.js")}
            }
            (package / "discards.js").write_text(
                f"window.DISCARDS = {json.dumps(payload, ensure_ascii=False)};\n",
                encoding="utf-8",
            )
        report["packages"][str(export)] = {
            "dataset_id": dsid or None,
            "discarded": len(drops),
            "gold_hits": gold_hits,
        }
    write_jsonl(OUT, [by_id[key] for key in sorted(by_id)])
    gold_total = sum(1 for rec in by_id.values() if rec.get("gold"))
    report["gold_discarded"] = gold_total
    report["remaining_t2_if_all_applied"] = 80 - sum(
        1
        for rec in by_id.values()
        if rec.get("gold") and rec["id"] in {str(r.get("id")) for r in read_jsonl(GOLD_T2)}
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if gold_total:
        print(
            f"warning: {gold_total} discarded items are frozen gold; "
            "the 15/80 gate will stay closed until they are replaced"
        )


if __name__ == "__main__":
    main()
