#!/usr/bin/env python3
"""Rewrite LayoutState dimension text to match CadView display.

Old states measured along the feature edge (85.09 / 84.14). CadView projects
onto dimAngle (85 / 4.75). Rebuild text from OSS CadAlignedDimension.dimAngle
and re-render annotate_t1 / annotate_t2 PNGs.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from fetch_production_dataset import (  # noqa: E402
    aligned_measurement,
    estimate_text,
    format_dim_value,
)
from layout_agent.evalset import render_preview  # noqa: E402
from layout_agent.models import LayoutState, Point  # noqa: E402


def _eid(obj: Any) -> Optional[int]:
    if obj is None:
        return None
    if isinstance(obj, dict) and "id" in obj:
        return int(obj["id"])
    if isinstance(obj, (int, float)):
        return int(obj)
    return None


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


def load_dim_catalog() -> Dict[Tuple[str, str], Dict[str, Any]]:
    catalog: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for p in sorted((ROOT / "data" / "raw").glob("solution_*/solution_doc.json")):
        sid = p.parent.name.replace("solution_", "")
        doc = json.loads(p.read_text(encoding="utf-8"))
        by: Dict[str, List[Dict[str, Any]]] = {}
        for el in doc.get("doc") or []:
            by.setdefault(str(el.get("_ctor_") or "Unknown"), []).append(el)
        for el in by.get("CadAlignedDimension", []):
            uid = str(el.get("uuid") or "")
            if not uid:
                continue
            ov = el.get("override") or {}
            text_ov = ov.get("text") if isinstance(ov.get("text"), dict) else {}
            try:
                ang = float(el.get("dimAngle") if el.get("dimAngle") is not None else 0)
            except (TypeError, ValueError):
                ang = 0.0
            dimdec = int(text_ov.get("dimdec") if text_ov.get("dimdec") is not None else 2)
            catalog[(sid, uid)] = {
                "kind": "linear",
                "override": ov,
                "dim_angle": ang,
                "dimdec": dimdec,
                "dimrnd": float(text_ov.get("dimrnd") or 0),
                "measurementType": str(el.get("measurementType") or "world"),
            }
        for el in by.get("CadRadialDimension", []):
            uid = str(el.get("uuid") or "")
            if not uid:
                continue
            ov = el.get("override") or {}
            text_ov = ov.get("text") if isinstance(ov.get("text"), dict) else {}
            pu = ov.get("primaryUnit") if isinstance(ov.get("primaryUnit"), dict) else {}
            cx, cy = _v2(el.get("center"))
            hx, hy = _v2(el.get("chordPoint"))
            catalog[(sid, uid)] = {
                "kind": "radial",
                "override": ov,
                "prefix": pu.get("dimPrefix") or "R",
                "suffix": pu.get("dimSuffix") or "",
                "dimdec": int(text_ov.get("dimdec") if text_ov.get("dimdec") is not None else 2),
                "radius": math.hypot(hx - cx, hy - cy),
            }
        print(f"catalog {sid}: {sum(1 for k in catalog if k[0]==sid)} dims")
    return catalog


def repair_state(raw: Dict[str, Any], catalog: Dict[Tuple[str, str], Dict[str, Any]]) -> int:
    src = raw.get("source") or {}
    sid = str(src.get("solution_id") or "")
    n = 0
    for lin in raw.get("linears") or []:
        meta = catalog.get((sid, str(lin.get("id") or "")))
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
        old = str((lin.get("text") or {}).get("content") or "")
        lin["value"] = value
        lin["dim_angle"] = meta["dim_angle"]
        text = dict(lin.get("text") or {})
        dim_scale = float(lin.get("dim_scale") or 1.0)
        lin["text"] = estimate_text(meta.get("override") or {"text": {"dimtxt": text.get("dimtxt") or 3.5}}, content, dim_scale)
        if old != content:
            n += 1
    for a in list(raw.get("diameters") or []) + list(raw.get("radials") or []):
        meta = catalog.get((sid, str(a.get("id") or "")))
        if not meta or meta.get("kind") != "radial":
            continue
        content = f"{meta['prefix']}{format_dim_value(float(meta['radius']), int(meta['dimdec']))}{meta['suffix']}"
        old = str((a.get("text") or {}).get("content") or "")
        dim_scale = float(a.get("dim_scale") or 1.0)
        text = dict(a.get("text") or {})
        a["text"] = estimate_text(meta.get("override") or {"text": {"dimtxt": text.get("dimtxt") or 3.5}}, content, dim_scale)
        if old != content:
            n += 1
    return n


def patch_states(catalog: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    roots = [
        ROOT / "data" / "evalsys" / "pairs_perturb" / "states",
        ROOT / "data" / "datasets",
    ]
    files: List[Path] = []
    for r in roots:
        if r.is_dir():
            files.extend(r.rglob("*.state.json"))
    changed_files = 0
    changed_dims = 0
    for p in files:
        raw = json.loads(p.read_text(encoding="utf-8"))
        n = repair_state(raw, catalog)
        if n:
            p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            changed_files += 1
            changed_dims += n
    print(f"patched {changed_files} state files, {changed_dims} dim texts")


def load_tasks(js_path: Path) -> List[Dict[str, Any]]:
    text = js_path.read_text(encoding="utf-8")
    prefix = "window.TASKS = "
    i = text.find(prefix)
    return json.loads(text[i + len(prefix) :].rstrip().rstrip(";"))


def rerender_pack(pack: Path) -> None:
    tasks = load_tasks(pack / "tasks.js")
    rend = pack / "renders"
    n = 0
    for t in tasks:
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
    print(f"re-rendered {n} images in {pack.name}")


def main() -> None:
    catalog = load_dim_catalog()
    patch_states(catalog)
    for name in ("annotate_t1", "annotate_t2"):
        rerender_pack(ROOT / "data" / "evalsys" / name)


if __name__ == "__main__":
    main()
