#!/usr/bin/env python3
"""Ingest colleague left/right drawing pairs into a frozen-test + growing-train registry.

Source project:
  https://v3.designorder.cn/shanshe-enterprise/project/2099378950165704704?tenantId=3

Left drawing = defective layout, right drawing = perfect layout. Views with no
dimensions, or with identical left/right poses, are dropped. The same filter is
applied to the legacy 中车 colleague pairs and to synthetic perturb pairs
(gold-referenced perturb pairs are kept).

Test view_keys are frozen. Later batches should be appended with:
  python scripts/evalset_ingest_human_pairs.py --append-only
which writes new modified pairs to train unless the view_key is already in test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_production_dataset import (  # noqa: E402
    DEFAULT_OSS,
    DEFAULT_TOKEN,
    _eid,
    _mat_tx,
    build_index,
    convert_view,
    download_oss_json,
    fetch_solution_meta,
    load_json_file,
)
from layout_agent.evalset.human_gold import read_jsonl, write_jsonl  # noqa: E402
from layout_agent.evalset.isolation import (  # noqa: E402
    forbidden_training_views,
    gold_views,
    write_forbidden_train_views,
)
from layout_agent.evalset.layout_diff import (  # noqa: E402
    annotation_count,
    layout_pose_fingerprint,
    layouts_identical,
    pose_delta,
)
from layout_agent.evalset.pairs import view_key  # noqa: E402
from layout_agent.evalset.split_human_pairs import SPLIT_N_TEST, SPLIT_SEED, random_test_keys  # noqa: E402
from layout_agent.evalset.train_variants import write_train_variants  # noqa: E402
from layout_agent.models import LayoutState  # noqa: E402
from layout_agent.perceive import ACTIVE_PALETTE  # noqa: E402

PROJECT_ID = "2099378950165704704"
TENANT = "3"
API_BASE = "https://api.designorder.cn/designBackend"
DRAWING_HOST = "https://drawing.designorder.cn"
ROW_TOL = 200.0
SEED = 17
EVALSYS = ROOT / "data" / "evalsys"
RAW_ROOT = ROOT / "data" / "raw"
NEW_DS = ROOT / "data" / "datasets" / "prod_colleague_pairs_enterprise_t3"
LEGACY_DS = ROOT / "data" / "datasets" / "prod_colleague_pairs_598848"
HUMAN_PAIRS = EVALSYS / "human_pairs"
PERTURB_MANIFEST = EVALSYS / "pairs_perturb" / "manifest.jsonl"
GOLD_DIR = EVALSYS / "human_gold"
REVIEW_ACTIONS = HUMAN_PAIRS / "review_actions.json"


def _headers() -> Dict[str, str]:
    return {
        "x-access-token": DEFAULT_TOKEN,
        "tenant-Id": TENANT,
        "content-type": "application/json;charset=UTF-8",
    }


def _safe(uuid: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in uuid)


def _rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def rel_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()).as_posix())
    except ValueError:
        return str(resolved.as_posix())


def load_state(path: Path) -> LayoutState:
    return LayoutState.model_validate(json.loads(path.read_text(encoding="utf-8")))


def list_project_solutions() -> List[Dict[str, Any]]:
    import httpx

    url = f"{API_BASE.rstrip('/')}/doMistServer/esResource/findByCondition"
    body = {
        "includeAllChildren": True,
        "pid": PROJECT_ID,
        "resourceCode": "solution",
        "types": ["file"],
    }
    rows: List[Dict[str, Any]] = []
    with httpx.Client(timeout=60.0) as client:
        for page in range(1, 8):
            resp = client.post(
                url,
                params={"tenantId": TENANT, "pageNumber": page, "pageSize": 50},
                json=body,
                headers=_headers(),
            )
            resp.raise_for_status()
            payload = resp.json()
            chunk = payload.get("result") or []
            rows.extend(chunk)
            total = int(payload.get("total") or 0)
            if page * 50 >= total or not chunk:
                break
    return rows


def fetch_solution_raw(solution_id: str, reuse: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw_dir = RAW_ROOT / f"solution_{solution_id}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path = raw_dir / "solution_meta.json"
    doc_path = raw_dir / "solution_doc.json"
    if reuse and meta_path.exists() and doc_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        doc = load_json_file(doc_path)
        return meta, doc
    meta = fetch_solution_meta(solution_id, TENANT, API_BASE, DEFAULT_TOKEN)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    key = meta.get("solutionSolutionWebDocument")
    if not key:
        raise RuntimeError(f"{solution_id}: missing solutionSolutionWebDocument")
    doc = download_oss_json(DEFAULT_OSS, key, doc_path)
    return meta, doc


def drawing_order(by: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for drawing in by.get("CadDrawing", []):
        tx, ty = _mat_tx(drawing.get("matrix"))
        rows.append({"eid": _eid(drawing.get("id")), "tx": tx, "ty": ty})
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: (-r["ty"], r["tx"]))
    clusters: List[List[Dict[str, Any]]] = []
    cur = [ordered[0]]
    for row in ordered[1:]:
        if abs(row["ty"] - cur[0]["ty"]) <= ROW_TOL:
            cur.append(row)
        else:
            clusters.append(sorted(cur, key=lambda x: x["tx"]))
            cur = [row]
    clusters.append(sorted(cur, key=lambda x: x["tx"]))
    flat: List[Dict[str, Any]] = []
    for i, cluster in enumerate(clusters):
        if len(cluster) % 2:
            print(f"  WARN drawing row {i} has odd count {len(cluster)}")
        for j, drawing in enumerate(cluster):
            drawing["row"] = i
            drawing["col"] = j
            flat.append(drawing)
    return flat


def view_dim_count(vid: int, by: Dict[str, List[Dict[str, Any]]]) -> int:
    n = 0
    for ctor in (
        "CadAlignedDimension",
        "CadDiameterDimension",
        "CadRadialDimension",
        "CadChamferDimension",
    ):
        n += sum(
            1
            for el in by.get(ctor, [])
            if el.get("pid") == vid and not el.get("inVisibleFlag")
        )
    return n


def views_on_drawing(
    drawing_eid: int,
    by: Dict[str, List[Dict[str, Any]]],
    groups_by_drawing: Dict[int, set],
) -> Dict[str, Dict[str, Any]]:
    vg_ids = groups_by_drawing.get(drawing_eid, set())
    out: Dict[str, Dict[str, Any]] = {}
    for view in by.get("CadView", []):
        if _eid(view.get("pid")) not in vg_ids:
            continue
        if view.get("inVisibleFlag"):
            continue
        vid = _eid(view.get("id"))
        n = view_dim_count(vid, by)
        if n <= 0 or vid is None:
            continue
        out[str(view.get("uuid") or "")] = {"eid": vid, "view": view, "dims": n}
    out.pop("", None)
    return out


def build_sheet_pairs(by: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    groups_by_drawing: Dict[int, set] = defaultdict(set)
    for group in by.get("CadViewGroup", []):
        de, ge = _eid(group.get("pid")), _eid(group.get("id"))
        if de is not None and ge is not None:
            groups_by_drawing[de].add(ge)
    order = drawing_order(by)
    pairs: List[Dict[str, Any]] = []
    for i in range(0, len(order) - 1, 2):
        left, right = order[i], order[i + 1]
        lv = views_on_drawing(left["eid"], by, groups_by_drawing)
        rv = views_on_drawing(right["eid"], by, groups_by_drawing)
        for uuid in sorted(set(lv) & set(rv)):
            if lv[uuid]["eid"] == rv[uuid]["eid"]:
                continue
            pairs.append(
                {
                    "pair_index": i // 2,
                    "uuid": uuid,
                    "viewType": lv[uuid]["view"].get("viewType"),
                    "perturbed_eid": lv[uuid]["eid"],
                    "perfect_eid": rv[uuid]["eid"],
                    "left_drawing": left["eid"],
                    "right_drawing": right["eid"],
                    "left_dims_cad": lv[uuid]["dims"],
                    "right_dims_cad": rv[uuid]["dims"],
                }
            )
    return pairs


def convert_pair_views(
    rec: Dict[str, Any],
    by: Dict[str, List[Dict[str, Any]]],
    meta: Dict[str, Any],
    out_dir: Path,
) -> Optional[Dict[str, Any]]:
    palette = ACTIVE_PALETTE * 20
    by_eid = {_eid(view.get("id")): view for view in by.get("CadView", [])}
    snap = out_dir / "snapshots"
    snap.mkdir(parents=True, exist_ok=True)
    item = dict(rec)
    for role, eid_key in (("perturbed", "perturbed_eid"), ("perfect", "perfect_eid")):
        eid = rec[eid_key]
        view = by_eid.get(eid)
        if view is None:
            return None
        try:
            state = convert_view(view, by, meta, palette)
        except Exception as exc:  # noqa: BLE001
            print(f"  convert fail eid={eid} uuid={rec['uuid']}: {exc}")
            return None
        if state is None:
            return None
        safe = _safe(str(rec["uuid"]))
        state.state_id = f"view_{safe}_{eid}"
        state.source = dict(state.source or {})
        state.source["cad_view_id"] = str(eid)
        state.source["colleague_role"] = role
        rel = f"snapshots/{state.state_id}.state.json"
        (out_dir / rel).write_text(
            json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        item[f"{role}_state"] = rel
        item[f"{role}_state_id"] = state.state_id
        item[f"{role}_n_ann"] = annotation_count(state)
        item[f"{role}_pose"] = layout_pose_fingerprint(state)
    left = load_state(out_dir / item["perturbed_state"])
    right = load_state(out_dir / item["perfect_state"])
    item["identical"] = layouts_identical(left, right)
    item["pose_delta"] = pose_delta(left, right)
    item["n_ann"] = min(item["perturbed_n_ann"], item["perfect_n_ann"])
    return item


def swap_pair_sides(item: Dict[str, Any]) -> Dict[str, Any]:
    """Treat sheet-right as defective and sheet-left as perfect."""
    out = dict(item)
    pairs = (
        ("perturbed_eid", "perfect_eid"),
        ("perturbed_state", "perfect_state"),
        ("perturbed_state_id", "perfect_state_id"),
        ("perturbed_n_ann", "perfect_n_ann"),
        ("perturbed_pose", "perfect_pose"),
        ("state_a", "state_b"),
    )
    for left_key, right_key in pairs:
        if left_key in out or right_key in out:
            out[left_key], out[right_key] = out.get(right_key), out.get(left_key)
    out["sides_swapped"] = True
    return out


def review_view_keys(kind: str) -> Set[str]:
    if not REVIEW_ACTIONS.exists():
        return set()
    data = json.loads(REVIEW_ACTIONS.read_text(encoding="utf-8"))
    return {
        str(row.get("view_key") or "")
        for row in data.get(kind) or []
        if isinstance(row, dict) and row.get("view_key")
    }


def apply_review_overrides(
    pool: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply human train-review: swap sides, drop low-quality, hold out cross-view."""
    swap_keys = review_view_keys("swap")
    drop_keys = review_view_keys("drop")
    cross_keys = review_view_keys("cross_view")
    kept: List[Dict[str, Any]] = []
    extra_dropped: List[Dict[str, Any]] = []
    cross_rows: List[Dict[str, Any]] = []
    for rec in pool:
        vk = str(rec.get("view_key") or "")
        item = dict(rec)
        if vk in drop_keys:
            item["reason"] = "review_drop_low_quality"
            extra_dropped.append(item)
            continue
        if vk in cross_keys:
            item["held_out"] = "cross_view"
            item["reason"] = "cross_view_annotation_mismatch"
            item["split"] = "cross_view"
            cross_rows.append(item)
            continue
        if vk in swap_keys and not item.get("sides_swapped"):
            item = swap_pair_sides(item)
        kept.append(item)
    return kept, extra_dropped, cross_rows


def relabel_swapped_snapshots(item: Dict[str, Any]) -> None:
    if not item.get("sides_swapped"):
        return
    dataset = str(item.get("dataset") or "")
    ds_root = ROOT / dataset if dataset else NEW_DS
    mapping = (
        ("perturbed", item.get("state_a") or item.get("perturbed_state")),
        ("perfect", item.get("state_b") or item.get("perfect_state")),
    )
    for role, rel in mapping:
        if not rel:
            continue
        path = Path(str(rel))
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists() and dataset:
            path = ds_root / Path(str(rel)).name
        if not path.exists():
            continue
        snap = json.loads(path.read_text(encoding="utf-8"))
        snap.setdefault("source", {})["colleague_role"] = role
        path.write_text(json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pair_record(
    *,
    pair_id: str,
    vkey: str,
    source: str,
    solution_id: str,
    tenant_id: str,
    dataset: str,
    perturbed_state: str,
    perfect_state: str,
    view_type: str,
    pose_delta_n: int,
    n_ann: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rec = {
        "pair_id": pair_id,
        "view_key": vkey,
        "source": source,
        "solution_id": solution_id,
        "tenant_id": tenant_id,
        "dataset": dataset,
        "state_a": perturbed_state,
        "state_b": perfect_state,
        "label": {"prefer": "b"},
        "view_type": view_type,
        "pose_delta": pose_delta_n,
        "n_ann": n_ann,
        "quality": pose_delta_n * 10 + min(int(n_ann), 20),
    }
    if extra:
        rec.update(extra)
    return rec


def ingest_new_solutions(reuse_raw: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    NEW_DS.mkdir(parents=True, exist_ok=True)
    solutions = list_project_solutions()
    catalog_path = RAW_ROOT / f"project_{PROJECT_ID}_solutions.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(solutions, ensure_ascii=False, indent=2), encoding="utf-8")
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
            item = convert_pair_views(rec, by, meta, NEW_DS)
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
            if vkey in review_view_keys("swap"):
                item = swap_pair_sides(item)
                for role, key in (("perturbed", "perturbed_state"), ("perfect", "perfect_state")):
                    snap_path = NEW_DS / item[key]
                    snap = json.loads(snap_path.read_text(encoding="utf-8"))
                    snap.setdefault("source", {})["colleague_role"] = role
                    snap_path.write_text(
                        json.dumps(snap, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            extra = {
                "drawing_url": f"{DRAWING_HOST}/?solutionId={sid}&tenantId={TENANT}",
                "solution_name": name,
                "perturbed_eid": item["perturbed_eid"],
                "perfect_eid": item["perfect_eid"],
            }
            if item.get("sides_swapped"):
                extra["sides_swapped"] = True
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
            pair_id = f"ent_{_safe(str(rec['uuid']))}_{sid[-6:]}_{rec['pair_index']}"
            record = pair_record(
                pair_id=pair_id,
                vkey=vkey,
                source="colleague_enterprise_t3",
                solution_id=sid,
                tenant_id=TENANT,
                dataset=str(NEW_DS.relative_to(ROOT).as_posix()),
                perturbed_state=str((NEW_DS / item["perturbed_state"]).as_posix()),
                perfect_state=str((NEW_DS / item["perfect_state"]).as_posix()),
                view_type=str(item.get("viewType") or ""),
                pose_delta_n=int(item["pose_delta"]),
                n_ann=int(item["n_ann"]),
                extra=extra,
            )
            record["state_a"] = rel_to_root(NEW_DS / item["perturbed_state"])
            record["state_b"] = rel_to_root(NEW_DS / item["perfect_state"])
            kept.append(record)
            ds_item = dict(item)
            ds_item["pair_id"] = pair_id
            ds_item["view_key"] = vkey
            ds_item["solution_id"] = sid
            ds_pairs.append(ds_item)
    (NEW_DS / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "prod_colleague_pairs_enterprise_t3",
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
    print(f"new kept {len(kept)} dropped {len(dropped)}")
    return kept, dropped


def ingest_legacy_colleague() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    man_path = LEGACY_DS / "manifest.json"
    if not man_path.exists():
        return [], []
    man = json.loads(man_path.read_text(encoding="utf-8"))
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for rec in man.get("pairs") or []:
        left_rel = rec.get("perturbed_state")
        right_rel = rec.get("perfect_state")
        left_path = LEGACY_DS / str(left_rel)
        right_path = LEGACY_DS / str(right_rel)
        if not left_path.exists() or not right_path.exists():
            dropped.append({"pair_id": rec.get("pair_id"), "reason": "missing_state"})
            continue
        left = load_state(left_path)
        right = load_state(right_path)
        vkey = view_key(left)
        extra = {
            "drawing_url": str(man.get("drawing_url") or ""),
            "perturbed_eid": rec.get("perturbed_eid"),
            "perfect_eid": rec.get("perfect_eid"),
        }
        if annotation_count(left) <= 0 or annotation_count(right) <= 0:
            dropped.append({"view_key": vkey, "reason": "no_annotations", **extra})
            continue
        if layouts_identical(left, right):
            dropped.append(
                {
                    "view_key": vkey,
                    "pair_id": rec.get("pair_id"),
                    "reason": "identical_pose",
                    "pose_delta": 0,
                    **extra,
                }
            )
            continue
        record = pair_record(
            pair_id=str(rec.get("pair_id") or f"colleague_{rec.get('uuid')}"),
            vkey=vkey,
            source="colleague_zhongche",
            solution_id=str(man.get("solution_id") or ""),
            tenant_id=str(man.get("tenant_id") or "250"),
            dataset=str(LEGACY_DS.relative_to(ROOT).as_posix()),
            perturbed_state=str(left_path.as_posix()),
            perfect_state=str(right_path.as_posix()),
            view_type=str(rec.get("viewType") or ""),
            pose_delta_n=pose_delta(left, right),
            n_ann=min(annotation_count(left), annotation_count(right)),
            extra=extra,
        )
        record["state_a"] = rel_to_root(left_path)
        record["state_b"] = rel_to_root(right_path)
        kept.append(record)
    print(f"legacy colleague kept {len(kept)} dropped {len(dropped)}")
    return kept, dropped


def gold_protected_pair_ids() -> Set[str]:
    ids: Set[str] = set()
    for name in ("t1_tasks.jsonl", "t2_tasks.jsonl"):
        path = GOLD_DIR / name
        if not path.exists():
            continue
        for row in read_jsonl(path):
            for key in ("id", "pair_id"):
                if row.get(key):
                    ids.add(str(row[key]))
    return ids


def filter_perturb_pairs() -> Dict[str, int]:
    if not PERTURB_MANIFEST.exists():
        return {"kept": 0, "dropped": 0, "protected": 0}
    rows = read_jsonl(PERTURB_MANIFEST)
    protected = gold_protected_pair_ids()
    keep: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    n_protected = 0
    root = PERTURB_MANIFEST.parent
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        a_path = root / str(row.get("state_a") or "")
        b_path = root / str(row.get("state_b") or "")
        identical = False
        if a_path.exists() and b_path.exists():
            try:
                identical = layouts_identical(load_state(a_path), load_state(b_path))
            except Exception:  # noqa: BLE001
                identical = False
        if identical and pair_id in protected:
            n_protected += 1
            keep.append(row)
            continue
        if identical:
            dropped.append(
                {
                    "pair_id": pair_id,
                    "view_key": row.get("view_key"),
                    "reason": "identical_pose",
                    "source": "perturb",
                    "split": row.get("split"),
                }
            )
            continue
        keep.append(row)
    backup = PERTURB_MANIFEST.with_suffix(".jsonl.bak_identical_filter")
    if not backup.exists():
        backup.write_text(PERTURB_MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    write_jsonl(PERTURB_MANIFEST, keep)
    write_jsonl(HUMAN_PAIRS / "dropped_perturb_identical.jsonl", dropped)
    print(
        f"perturb kept {len(keep)} dropped {len(dropped)} gold-protected-identical {n_protected}"
    )
    return {"kept": len(keep), "dropped": len(dropped), "protected": n_protected}


def test_size(n: int) -> int:
    if n <= 0:
        return 0
    leave_train = min(30, max(10, n // 10)) if n > 20 else max(1, n // 5)
    return max(1, min(n - leave_train, 180))


def choose_test(pool: Sequence[Dict[str, Any]]) -> Set[str]:
    """Uniform random freeze — not the high-delta subset."""
    return set(
        random_test_keys(
            [str(rec.get("view_key") or "") for rec in pool],
            n_test=SPLIT_N_TEST,
            seed=SPLIT_SEED,
        )
    )


def dedupe_view_keys(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the highest-quality pair per view_key."""
    best: Dict[str, Dict[str, Any]] = {}
    for rec in rows:
        key = str(rec.get("view_key") or "")
        if not key:
            continue
        prev = best.get(key)
        if prev is None or int(rec.get("quality") or 0) > int(prev.get("quality") or 0):
            best[key] = rec
        elif int(rec.get("quality") or 0) == int(prev.get("quality") or 0):
            if _rank(str(rec.get("pair_id"))) < _rank(str(prev.get("pair_id"))):
                best[key] = rec
    return [best[k] for k in sorted(best)]


def write_registry(
    pool: List[Dict[str, Any]],
    dropped: List[Dict[str, Any]],
    perturb_stats: Dict[str, int],
    append_only: bool,
) -> Dict[str, Any]:
    HUMAN_PAIRS.mkdir(parents=True, exist_ok=True)
    registry_path = HUMAN_PAIRS / "registry.json"
    existing_test: Set[str] = set()
    if registry_path.exists():
        old = json.loads(registry_path.read_text(encoding="utf-8"))
        existing_test = set(old.get("test_view_keys") or [])

    gold = gold_views(GOLD_DIR)
    pool = [rec for rec in pool if rec["view_key"] not in gold]
    pool = dedupe_view_keys(pool)
    pool, review_dropped, cross_rows = apply_review_overrides(pool)
    dropped.extend(review_dropped)
    for rec in pool:
        if rec.get("sides_swapped"):
            relabel_swapped_snapshots(rec)

    if append_only and existing_test:
        test_keys = set(existing_test)
    else:
        test_keys = choose_test(pool)

    test_rows = [rec for rec in pool if rec["view_key"] in test_keys]
    train_rows = [rec for rec in pool if rec["view_key"] not in test_keys]
    for rec in test_rows:
        rec["split"] = "test"
    for rec in train_rows:
        rec["split"] = "train"

    write_jsonl(HUMAN_PAIRS / "test.jsonl", test_rows)
    write_jsonl(HUMAN_PAIRS / "train.jsonl", train_rows)
    write_jsonl(HUMAN_PAIRS / "dropped.jsonl", dropped)
    write_jsonl(HUMAN_PAIRS / "all_modified.jsonl", test_rows + train_rows)
    write_jsonl(HUMAN_PAIRS / "cross_view.jsonl", cross_rows)

    now = datetime.now(timezone.utc).astimezone().isoformat()
    registry = {
        "schema_version": 1,
        "frozen_at": now,
        "project_id": PROJECT_ID,
        "tenant_id": TENANT,
        "project_url": (
            "https://v3.designorder.cn/shanshe-enterprise/project/"
            f"{PROJECT_ID}?tenantId={TENANT}&app=shanshe-enterprise"
        ),
        "policy": {
            "test_is_frozen": True,
            "new_pairs_go_to": "train",
            "skip_if_view_key_in_test": True,
            "drop_identical_sides": True,
            "drop_unannotated": True,
            "seed": SPLIT_SEED,
            "split_method": "random",
            "n_test": SPLIT_N_TEST,
            "label": "prefer perfect/right (b) over defective/left (a)",
        },
        "counts": {
            "modified_pairs": len(pool),
            "test_pairs": len(test_rows),
            "train_pairs": len(train_rows),
            "dropped": len(dropped),
            "cross_view": len(cross_rows),
            "test_solutions": len({r["solution_id"] for r in test_rows}),
            "train_solutions": len({r["solution_id"] for r in train_rows}),
            "perturb_filter": perturb_stats,
        },
        "test_view_keys": sorted(test_keys),
        "train_view_keys": sorted({r["view_key"] for r in train_rows}),
        "how_to_append": (
            "python scripts/evalset_ingest_human_pairs.py --append-only "
            "# new modified pairs go to train.jsonl; test.jsonl stays frozen"
        ),
    }
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    write_train_variants(ROOT)

    forbidden = forbidden_training_views(ROOT)
    write_forbidden_train_views(ROOT, forbidden)
    return registry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-raw", action="store_true")
    ap.add_argument(
        "--append-only",
        action="store_true",
        help="keep frozen test_view_keys; send new modified pairs to train",
    )
    ap.add_argument("--skip-fetch", action="store_true", help="only reshuffle existing snapshots")
    args = ap.parse_args()

    HUMAN_PAIRS.mkdir(parents=True, exist_ok=True)
    dropped: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []

    legacy_kept, legacy_dropped = ingest_legacy_colleague()
    kept.extend(legacy_kept)
    dropped.extend(legacy_dropped)

    if args.skip_fetch:
        man_path = NEW_DS / "manifest.json"
        if man_path.exists():
            man = json.loads(man_path.read_text(encoding="utf-8"))
            for item in man.get("pairs") or []:
                left = load_state(NEW_DS / item["perturbed_state"])
                right = load_state(NEW_DS / item["perfect_state"])
                if item.get("identical") or layouts_identical(left, right):
                    dropped.append({"view_key": item.get("view_key"), "reason": "identical_pose"})
                    continue
                kept.append(
                    pair_record(
                        pair_id=str(item["pair_id"]),
                        vkey=str(item["view_key"]),
                        source="colleague_enterprise_t3",
                        solution_id=str(item.get("solution_id") or ""),
                        tenant_id=TENANT,
                        dataset=str(NEW_DS.relative_to(ROOT).as_posix()),
                        perturbed_state=str((NEW_DS / item["perturbed_state"]).as_posix()),
                        perfect_state=str((NEW_DS / item["perfect_state"]).as_posix()),
                        view_type=str(item.get("viewType") or ""),
                        pose_delta_n=int(item.get("pose_delta") or pose_delta(left, right)),
                        n_ann=int(item.get("n_ann") or 0),
                    )
                )
    else:
        new_kept, new_dropped = ingest_new_solutions(reuse_raw=args.reuse_raw or args.append_only)
        kept.extend(new_kept)
        dropped.extend(new_dropped)

    perturb_stats = filter_perturb_pairs()
    registry = write_registry(kept, dropped, perturb_stats, append_only=args.append_only)
    print(json.dumps(registry["counts"], ensure_ascii=False, indent=2))
    print("registry ->", HUMAN_PAIRS / "registry.json")


if __name__ == "__main__":
    main()
