#!/usr/bin/env python3
"""Patch LayoutState with CAD roughness + dim deviations, then re-render previews.

Existing states dropped CadUserBlock(roughness_down) and override.tolerance.
Rebuild those fields from OSS solution_doc.json without moving perturbed poses.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from fetch_production_dataset import (  # noqa: E402
    _eid,
    aligned_measurement,
    convert_roughness_block,
    estimate_text,
    format_dim_value,
)
from layout_agent.evalset import render_preview  # noqa: E402
from layout_agent.models import LayoutState, Point  # noqa: E402


def load_catalogs() -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    dims: Dict[Tuple[str, str], Dict[str, Any]] = {}
    roughness: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for p in sorted((ROOT / "data" / "raw").glob("solution_*/solution_doc.json")):
        sid = p.parent.name.replace("solution_", "")
        doc = json.loads(p.read_text(encoding="utf-8"))["doc"]
        views: Dict[Any, Dict[str, Any]] = {}
        by_pid: Dict[Any, List[Dict[str, Any]]] = {}
        for el in doc:
            ctor = el.get("_ctor_")
            if ctor == "CadView":
                views[_eid(el.get("id"))] = el
            pid = _eid(el.get("pid"))
            if pid is not None:
                by_pid.setdefault(pid, []).append(el)
        for vid, view in views.items():
            vu = str(view.get("uuid") or vid or "")
            vmat = view.get("matrix")
            rgs = []
            for el in by_pid.get(vid, []):
                ctor = el.get("_ctor_")
                uid = str(el.get("uuid") or "")
                ov = el.get("override") or {}
                if ctor == "CadAlignedDimension" and uid:
                    text_ov = ov.get("text") if isinstance(ov.get("text"), dict) else {}
                    try:
                        ang = float(el.get("dimAngle") if el.get("dimAngle") is not None else 0)
                    except (TypeError, ValueError):
                        ang = 0.0
                    dims[(sid, uid)] = {
                        "kind": "linear",
                        "override": ov,
                        "dim_angle": ang,
                        "dimdec": int(text_ov.get("dimdec") if text_ov.get("dimdec") is not None else 2),
                        "dimrnd": float(text_ov.get("dimrnd") or 0),
                        "measurementType": str(el.get("measurementType") or "world"),
                    }
                elif ctor == "CadDiameterDimension" and uid:
                    pu = ov.get("primaryUnit") if isinstance(ov.get("primaryUnit"), dict) else {}
                    text_ov = ov.get("text") if isinstance(ov.get("text"), dict) else {}
                    hole = el.get("relatedHoleDetailedInfo") or {}
                    dims[(sid, uid)] = {
                        "kind": "diameter",
                        "override": ov,
                        "prefix": pu.get("dimPrefix") or "⌀",
                        "suffix": pu.get("dimSuffix") or "",
                        "dimdec": int(text_ov.get("dimdec") if text_ov.get("dimdec") is not None else 2),
                        "diameter": hole.get("diameter_value"),
                    }
                elif ctor == "CadRadialDimension" and uid:
                    pu = ov.get("primaryUnit") if isinstance(ov.get("primaryUnit"), dict) else {}
                    text_ov = ov.get("text") if isinstance(ov.get("text"), dict) else {}
                    cx, cy = _v2(el.get("center"))
                    hx, hy = _v2(el.get("chordPoint"))
                    dims[(sid, uid)] = {
                        "kind": "radial",
                        "override": ov,
                        "prefix": pu.get("dimPrefix") or "R",
                        "suffix": pu.get("dimSuffix") or "",
                        "dimdec": int(text_ov.get("dimdec") if text_ov.get("dimdec") is not None else 2),
                        "radius": math.hypot(hx - cx, hy - cy),
                    }
                rg = convert_roughness_block(el, vmat)
                if rg is not None:
                    rgs.append(rg.model_dump(mode="json"))
            if rgs:
                roughness[(sid, vu)] = rgs
        print(f"catalog {sid}: {sum(1 for k in dims if k[0]==sid)} dims, "
              f"{sum(len(v) for k,v in roughness.items() if k[0]==sid)} roughness")
    return dims, roughness


def _v2(obj: Any) -> Tuple[float, float]:
    if obj is None:
        return 0.0, 0.0
    if isinstance(obj, dict):
        data = obj.get("data", obj)
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            return float(data[0]), float(data[1])
        return float(obj.get("x", 0)), float(obj.get("y", 0))
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return float(obj[0]), float(obj[1])
    return 0.0, 0.0


def repair_state(
    raw: Dict[str, Any],
    dims: Dict[Tuple[str, str], Dict[str, Any]],
    roughness: Dict[Tuple[str, str], List[Dict[str, Any]]],
) -> int:
    src = raw.get("source") or {}
    sid = str(src.get("solution_id") or "")
    vu = str(src.get("view_uuid") or "")
    n = 0
    for lin in raw.get("linears") or []:
        meta = dims.get((sid, str(lin.get("id") or "")))
        if not meta or meta.get("kind") != "linear":
            continue
        p1 = lin.get("p1") or {}
        p2 = lin.get("p2") or {}
        value = aligned_measurement(
            Point(x=float(p1.get("x", 0)), y=float(p1.get("y", 0))),
            Point(x=float(p2.get("x", 0)), y=float(p2.get("y", 0))),
            float(meta["dim_angle"]),
            measurement_type=str(meta.get("measurementType") or "world"),
            dimrnd=float(meta.get("dimrnd") or 0),
        )
        content = format_dim_value(value, int(meta["dimdec"]))
        dim_scale = float(lin.get("dim_scale") or 1.0)
        new_text = estimate_text(meta.get("override") or {}, content, dim_scale)
        old = lin.get("text") or {}
        lin["value"] = value
        lin["dim_angle"] = meta["dim_angle"]
        lin["text"] = new_text
        if old.get("content") != new_text.get("content") or old.get("tolerance") != new_text.get("tolerance"):
            n += 1
    for a in list(raw.get("diameters") or []) + list(raw.get("radials") or []):
        meta = dims.get((sid, str(a.get("id") or "")))
        if not meta or meta.get("kind") not in ("diameter", "radial"):
            continue
        if meta["kind"] == "diameter":
            diam = meta.get("diameter")
            if diam is None:
                content = f"{meta['prefix']}{meta['suffix']}" or "⌀"
            else:
                content = f"{meta['prefix']}{format_dim_value(float(diam), int(meta['dimdec']))}{meta['suffix']}"
        else:
            content = f"{meta['prefix']}{format_dim_value(float(meta['radius']), int(meta['dimdec']))}{meta['suffix']}"
        dim_scale = float(a.get("dim_scale") or 1.0)
        new_text = estimate_text(meta.get("override") or {}, content, dim_scale)
        old = a.get("text") or {}
        a["text"] = new_text
        if old.get("content") != new_text.get("content") or old.get("tolerance") != new_text.get("tolerance"):
            n += 1
    rgs = roughness.get((sid, vu))
    if rgs is not None:
        prev = raw.get("roughness") or []
        raw["roughness"] = rgs
        if prev != rgs:
            n += 1
    return n


def patch_states(
    dims: Dict[Tuple[str, str], Dict[str, Any]],
    roughness: Dict[Tuple[str, str], List[Dict[str, Any]]],
    view_uuid: Optional[str] = None,
) -> List[Path]:
    roots = [
        ROOT / "data" / "evalsys" / "pairs_perturb" / "states",
        ROOT / "data" / "datasets" / "prod_070-3579",
    ]
    files: List[Path] = []
    for r in roots:
        if r.is_dir():
            files.extend(r.rglob("*.state.json"))
    changed: List[Path] = []
    n_dims = 0
    for p in files:
        if view_uuid and view_uuid not in p.name:
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        n = repair_state(raw, dims, roughness)
        if n:
            p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            changed.append(p)
            n_dims += n
    print(f"patched {len(changed)} state files ({n_dims} field updates)")
    return changed


def load_tasks(js_path: Path) -> List[Dict[str, Any]]:
    text = js_path.read_text(encoding="utf-8")
    prefix = "window.TASKS = "
    i = text.find(prefix)
    return json.loads(text[i + len(prefix) :].rstrip().rstrip(";"))


def rerender_pack(pack: Path, view_uuid: Optional[str] = None, index: Optional[int] = None) -> None:
    tasks = load_tasks(pack / "tasks.js")
    rend = pack / "renders"
    n = 0
    for i, t in enumerate(tasks):
        if index is not None and i != index:
            continue
        if view_uuid and view_uuid not in str(t.get("id") or "") and view_uuid not in str(t.get("view_uuid") or ""):
            continue
        if t.get("type") == "prefer":
            pairs = ((t.get("state_a"), t.get("render_a")), (t.get("state_b"), t.get("render_b")))
        else:
            pairs = ((t.get("state"), t.get("render")),)
        for sp, rp in pairs:
            if not sp or not rp:
                continue
            src = Path(sp)
            dst = rend / Path(rp).name
            if not src.exists():
                print(f"missing state {src}")
                continue
            st = LayoutState.model_validate(json.loads(src.read_text(encoding="utf-8")))
            render_preview(st, dst)
            n += 1
            print(f"rendered {dst}")
    print(f"re-rendered {n} images in {pack.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="", help="Only patch/render this view uuid (e.g. 1DgsX4Uo)")
    ap.add_argument("--t2-index", type=int, default=None, help="0-based T2 task index to render")
    ap.add_argument("--all-packs", action="store_true", help="Re-render annotate_t1 and annotate_t2")
    args = ap.parse_args()
    dims, roughness = load_catalogs()
    patch_states(dims, roughness, view_uuid=args.view or None)
    t2 = ROOT / "data" / "evalsys" / "annotate_t2"
    if args.t2_index is not None:
        rerender_pack(t2, index=args.t2_index)
        return
    if args.view:
        rerender_pack(t2, view_uuid=args.view)
        t1 = ROOT / "data" / "evalsys" / "annotate_t1"
        if t1.exists():
            rerender_pack(t1, view_uuid=args.view)
        return
    if args.all_packs:
        for name in ("annotate_t1", "annotate_t2"):
            pack = ROOT / "data" / "evalsys" / name
            if pack.exists():
                rerender_pack(pack)


if __name__ == "__main__":
    main()
