#!/usr/bin/env python3
"""Fetch a production solution and convert CadView(+dims) → Layout Agent dataset.

Uses the same production APIs as 2d_service SolutionLoadUtil:
  GET  {base}/doMistServer/mistSolution/selectById?solutionId=...
  OSS  {ossPrefix}{solutionSolutionWebDocument}

Default target:
  https://drawing.designorder.cn/?solutionId=2082016429787918336&tenantId=0
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.config import load_config
from layout_agent.cad_marks import (
    attach_tolerance,
    is_roughness_block,
    roughness_extent,
    roughness_value_from_block,
)
from layout_agent.models import (
    ColorLegend,
    ColorLegendEntry,
    LayoutState,
    LinearAnno,
    Obstacle,
    Point,
    RadialAnno,
    RasterMeta,
    RoughnessAnno,
)
from layout_agent.perceive import ACTIVE_PALETTE, Perceiver

DEFAULT_SOLUTION_ID = "2082016429787918336"
DEFAULT_TENANT_ID = "0"
DEFAULT_BASE = "https://api.designorder.cn/designBackend"
DEFAULT_OSS = "https://designorder.oss-cn-shanghai.aliyuncs.com/"
# Same service token used by 2d_service headless loaders (SolutionLoadUtil / NodeAppService)
DEFAULT_TOKEN = "BD580877-32E0-1FF1-CE5E-B1EE94202857"


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


def _mat_tx(matrix: Any) -> Tuple[float, float]:
    """Extract translation from Matrix4 dump (row-major with tx/ty in data[3][0:2])."""
    if not matrix:
        return 0.0, 0.0
    data = matrix.get("data") if isinstance(matrix, dict) else matrix
    if not data or len(data) < 4:
        return 0.0, 0.0
    row = data[3]
    return float(row[0]), float(row[1])


def _apply_mat(pt: Tuple[float, float], matrix: Any) -> Tuple[float, float]:
    """Apply translation-only Matrix4 (matches production dim dumps)."""
    tx, ty = _mat_tx(matrix)
    return pt[0] + tx, pt[1] + ty


def _to_view_local(
    sheet_pt: Tuple[float, float],
    view_matrix: Any,
) -> Point:
    vtx, vty = _mat_tx(view_matrix)
    return Point(x=sheet_pt[0] - vtx, y=sheet_pt[1] - vty)


def fetch_solution_meta(
    solution_id: str,
    tenant_id: str,
    base_url: str,
    token: str,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/doMistServer/mistSolution/selectById"
    headers = {
        "x-access-token": token,
        "tenant-Id": tenant_id,
        "content-type": "application/json;charset=UTF-8",
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(url, params={"solutionId": solution_id}, headers=headers)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"selectById failed: {payload}")
    rows = payload.get("result") or []
    if not rows:
        raise RuntimeError("selectById returned empty result")
    return rows[0]


def load_json_bytes(raw: bytes) -> Dict[str, Any]:
    """Parse JSON; tolerate gzip (OSS sometimes returns Content-Encoding: gzip as raw bytes)."""
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def load_json_file(path: Path) -> Dict[str, Any]:
    return load_json_bytes(path.read_bytes())


def download_oss_json(oss_prefix: str, key: str, dest: Path) -> Dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{oss_prefix.rstrip('/')}/{key.lstrip('/')}?t={int(datetime.now().timestamp()*1000)}"
    with httpx.Client(timeout=300.0) as client:
        resp = client.get(url)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return load_json_bytes(resp.content)


def build_index(doc: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for el in doc.get("doc") or []:
        ctor = el.get("_ctor_") or "Unknown"
        by.setdefault(ctor, []).append(el)
    return by


def view_obstacles(view_id: int, by_ctor: Dict[str, List[Dict[str, Any]]]) -> List[Obstacle]:
    obstacles: List[Obstacle] = []
    n = 0
    for ctor in ("CadLines", "CadPolylines", "CadArcs"):
        for el in by_ctor.get(ctor, []):
            if el.get("pid") != view_id:
                continue
            if el.get("inVisibleFlag"):
                continue
            # prefer visible projection geometry as silhouette obstacles
            if el.get("isProjectionVisible") is False and el.get("projectionType") == "HIDDEN_LINE":
                continue
            for i, item in enumerate(el.get("datas") or []):
                pts: List[Point] = []
                if item.get("type") == "line":
                    s = item.get("start") or [0, 0]
                    e = item.get("end") or [0, 0]
                    pts = [Point(x=float(s[0]), y=float(s[1])), Point(x=float(e[0]), y=float(e[1]))]
                elif item.get("type") == "polyline":
                    for v in item.get("vertexes") or []:
                        pts.append(Point(x=float(v[0]), y=float(v[1])))
                elif item.get("type") == "arc":
                    # Production CadArcs store angles in degrees (not radians).
                    c = item.get("center") or [0, 0]
                    r = float(item.get("radius") or 0)
                    a0_raw = float(item.get("startAngle") or 0)
                    a1_raw = float(item.get("endAngle") or 360)
                    if r <= 0:
                        continue
                    a0 = math.radians(a0_raw)
                    a1 = math.radians(a1_raw)
                    sweep = a1 - a0
                    while sweep > math.pi * 2:
                        sweep -= math.pi * 2
                    while sweep < -math.pi * 2:
                        sweep += math.pi * 2
                    if abs(sweep) < math.radians(0.5):
                        continue
                    steps = max(4, int(abs(sweep) / (math.pi / 12)))
                    for k in range(steps + 1):
                        t = a0 + sweep * k / steps
                        pts.append(Point(x=c[0] + r * math.cos(t), y=c[1] + r * math.sin(t)))
                if len(pts) >= 2:
                    n += 1
                    obstacles.append(
                        Obstacle(
                            id=f"proj_{ctor}_{_eid(el.get('id'))}_{i}",
                            kind="polyline",
                            raw_type=str(item.get("feature", {}).get("basicType") or ctor),
                            polyline=pts,
                            fill_role="edge",
                        )
                    )
    # also include view loop as outer silhouette
    return obstacles


def estimate_text(override: Dict[str, Any], content: str, dim_scale: float = 1.0) -> Dict[str, Any]:
    dimtxt = float(((override or {}).get("text") or {}).get("dimtxt") or 3.5)
    # CAD on-sheet text size ≈ dimtxt * dimScale (same as drawing-2d overrideToDim)
    scale = max(float(dim_scale or 1.0), 1e-6)
    th = dimtxt * scale
    width = max(th * 1.2, len(content) * th * 0.55)
    text = {"content": content, "width": width, "height": th * 1.15, "dimtxt": dimtxt, "dim_scale": scale}
    return attach_tolerance(text, override, scale)


def aligned_measurement(
    p1: Point,
    p2: Point,
    dim_angle_deg: Optional[float],
    *,
    measurement_type: str = "world",
    parent_scale: float = 1.0,
    dimrnd: float = 0.0,
) -> float:
    """Match CadAlignedDimensionUtil.getAlignedParam distance.

    Value = distance between projections of p1/p2 onto the dim-line direction (dimAngle),
    not the Euclidean edge length. E.g. slanted edge with dimAngle=0 → |Δx|, dimAngle=90 → |Δy|.
    """
    if dim_angle_deg is None:
        # CAD default dimAngle is 0° (horizontal dim line). Do not fall back to
        # the feature-edge direction — that yields the slant length (e.g. 85.09 /
        # 84.14) instead of the displayed projection (85 / 4.75).
        ang = 0.0
    else:
        ang = math.radians(float(dim_angle_deg))
    ux, uy = math.cos(ang), math.sin(ang)
    t1 = p1.x * ux + p1.y * uy
    t2 = p2.x * ux + p2.y * uy
    distance = abs(t2 - t1)
    if str(measurement_type).lower() == "reference" and parent_scale:
        distance /= float(parent_scale)
    rnd = float(dimrnd or 0.0)
    if rnd > 0:
        distance = round(distance / rnd) * rnd
    return distance


def format_dim_value(value: float, dimdec: int = 2) -> str:
    dec = max(0, int(dimdec))
    return f"{value:.{dec}f}".rstrip("0").rstrip(".")


def convert_roughness_block(el: Dict[str, Any], view_matrix: Any) -> Optional[RoughnessAnno]:
    if not is_roughness_block(el) or el.get("inVisibleFlag"):
        return None
    args = el.get("args") if isinstance(el.get("args"), dict) else {}
    insert = _to_view_local(_apply_mat(_v2(el.get("insertPt")), el.get("matrix")), view_matrix)
    try:
        style = int(args.get("roughnessStyle") or 1)
    except (TypeError, ValueError):
        style = 1
    try:
        rot = float(el.get("rotation") or 0.0)
    except (TypeError, ValueError):
        rot = 0.0
    uid = str(el.get("uuid") or f"RG_{_eid(el.get('id'))}")
    return RoughnessAnno(
        id=uid,
        insert=insert,
        rotation=rot,
        scale=float(el.get("scale") or 1.0),
        roughness_scale=float(args.get("roughnessScale") or 1.0),
        value=roughness_value_from_block(el),
        style=style,
        text_height=float(args.get("textHeight") or 2.5),
        triangle_side=float(args.get("triangleSideLength") or 2.89),
        prefix=str(args.get("roughnessTextPrefix") or "Rz"),
    )


def convert_view(
    view: Dict[str, Any],
    by_ctor: Dict[str, List[Dict[str, Any]]],
    solution_meta: Dict[str, Any],
    palette: List[str],
) -> Optional[LayoutState]:
    view_id = _eid(view.get("id"))
    if view_id is None:
        return None
    vmin = _v2(view.get("min"))
    vmax = _v2(view.get("max"))
    vmat = view.get("matrix")

    linears: List[LinearAnno] = []
    diameters: List[RadialAnno] = []
    radials: List[RadialAnno] = []
    legend = ColorLegend(entries={})
    color_i = 0

    def next_color() -> str:
        nonlocal color_i
        c = palette[color_i % len(palette)]
        color_i += 1
        return c

    for el in by_ctor.get("CadAlignedDimension", []):
        if el.get("pid") != view_id or el.get("inVisibleFlag"):
            continue
        p1s = _apply_mat(_v2(el.get("xLine1Point")), el.get("matrix"))
        p2s = _apply_mat(_v2(el.get("xLine2Point")), el.get("matrix"))
        dps = _apply_mat(_v2(el.get("dimLinePoint")), el.get("matrix"))
        tps = _apply_mat(_v2(el.get("textPosition") or el.get("dimLinePoint")), el.get("matrix"))
        p1 = _to_view_local(p1s, vmat)
        p2 = _to_view_local(p2s, vmat)
        point = _to_view_local(dps, vmat)
        text_point = _to_view_local(tps, vmat)
        # direction from feature to dim line
        mid = Point(x=(p1.x + p2.x) / 2, y=(p1.y + p2.y) / 2)
        dx, dy = point.x - mid.x, point.y - mid.y
        norm = math.hypot(dx, dy) or 1.0
        ov = el.get("override") or {}
        text_ov = ov.get("text") if isinstance(ov.get("text"), dict) else {}
        dim_angle = el.get("dimAngle")
        try:
            # CAD always has dimAngle; 0 = horizontal. Missing → 0, never "along edge".
            dim_angle_f = float(dim_angle) if dim_angle is not None else 0.0
        except (TypeError, ValueError):
            dim_angle_f = 0.0
        dim_scale = float(el.get("dimScale") or view.get("scale") or 1.0)
        parent_scale = 1.0
        # matrix scale * view.ref.scale (drawing-2d elementToDimParam)
        try:
            mdata = (el.get("matrix") or {}).get("data") or []
            if mdata and len(mdata) >= 2:
                parent_scale = abs(float(mdata[0][0])) or 1.0
        except (TypeError, ValueError, IndexError):
            parent_scale = 1.0
        vref = (view.get("ref") or {}) if isinstance(view.get("ref"), dict) else {}
        if vref.get("scale"):
            parent_scale *= float(vref["scale"])
        # Measure in CAD local coords (xLine + dimAngle), matching CadAlignedDimensionUtil.
        x1, y1 = _v2(el.get("xLine1Point"))
        x2, y2 = _v2(el.get("xLine2Point"))
        raw_p1 = Point(x=x1, y=y1)
        raw_p2 = Point(x=x2, y=y2)
        value = aligned_measurement(
            raw_p1,
            raw_p2,
            dim_angle_f,
            measurement_type=str(el.get("measurementType") or "world"),
            parent_scale=parent_scale,
            dimrnd=float(text_ov.get("dimrnd") or 0),
        )
        dimdec = int(text_ov.get("dimdec") if text_ov.get("dimdec") is not None else 2)
        content = format_dim_value(value, dimdec)
        color = next_color()
        uid = str(el.get("uuid") or f"L_{_eid(el.get('id'))}")
        anno = LinearAnno(
            id=uid,
            color=color,
            p1=p1,
            p2=p2,
            dir=Point(x=dx / norm, y=dy / norm),
            point=point,
            value=value,
            arrow_width=float(((ov.get("arrow") or {}).get("dimasz") or 2.5)) * dim_scale,
            text=estimate_text(ov, content, dim_scale),
            dim_angle=dim_angle_f,
            text_point=text_point,
            dim_scale=dim_scale,
            dimtih=bool(text_ov.get("dimtih", False)),
            dimtoh=bool(text_ov.get("dimtoh", False)),
        )
        linears.append(anno)
        legend.entries[color] = ColorLegendEntry(id=uid, category="linear")

    for el in by_ctor.get("CadDiameterDimension", []):
        if el.get("pid") != view_id or el.get("inVisibleFlag"):
            continue
        chord = _apply_mat(_v2(el.get("chordPoint")), el.get("matrix"))
        far = _apply_mat(_v2(el.get("farChordPoint")), el.get("matrix"))
        text_s = _apply_mat(_v2(el.get("textPosition")), el.get("matrix"))
        cx, cy = (chord[0] + far[0]) / 2, (chord[1] + far[1]) / 2
        anchor = _to_view_local((cx, cy), vmat)
        end = _to_view_local(text_s, vmat)
        mid = Point(x=(anchor.x + end.x) / 2, y=(anchor.y + end.y) / 2)
        ov = el.get("override") or {}
        hole = el.get("relatedHoleDetailedInfo") or {}
        diam = hole.get("diameter_value")
        pu = ov.get("primaryUnit") if isinstance(ov.get("primaryUnit"), dict) else {}
        prefix = pu.get("dimPrefix") or "⌀"
        suffix = pu.get("dimSuffix") or ""
        text_ov = ov.get("text") if isinstance(ov.get("text"), dict) else {}
        dimdec = int(text_ov.get("dimdec") if text_ov.get("dimdec") is not None else 2)
        if diam is not None:
            content = f"{prefix}{format_dim_value(float(diam), dimdec)}{suffix}"
        else:
            content = f"{prefix}{suffix}" if (prefix or suffix) else "⌀"
        color = next_color()
        uid = str(el.get("uuid") or f"D_{_eid(el.get('id'))}")
        dim_scale = float(el.get("dimScale") or view.get("scale") or 1.0)
        anno = RadialAnno(
            id=uid,
            category="diameter",
            color=color,
            anchor=anchor,
            path=[anchor, mid, end],
            text=estimate_text(el.get("override") or {}, content, dim_scale),
            dim_scale=dim_scale,
        )
        diameters.append(anno)
        legend.entries[color] = ColorLegendEntry(id=uid, category="diameter")

    for el in by_ctor.get("CadRadialDimension", []):
        if el.get("pid") != view_id or el.get("inVisibleFlag"):
            continue
        center_s = _apply_mat(_v2(el.get("center")), el.get("matrix"))
        text_s = _apply_mat(_v2(el.get("textPosition")), el.get("matrix"))
        chord_s = _apply_mat(_v2(el.get("chordPoint")), el.get("matrix"))
        anchor = _to_view_local(center_s, vmat)
        end = _to_view_local(text_s, vmat)
        via = _to_view_local(chord_s, vmat)
        ov = el.get("override") or {}
        text_ov = ov.get("text") if isinstance(ov.get("text"), dict) else {}
        pu = ov.get("primaryUnit") if isinstance(ov.get("primaryUnit"), dict) else {}
        prefix = pu.get("dimPrefix") or "R"
        suffix = pu.get("dimSuffix") or ""
        cx, cy = _v2(el.get("center"))
        hx, hy = _v2(el.get("chordPoint"))
        radius = math.hypot(hx - cx, hy - cy)
        dimdec = int(text_ov.get("dimdec") if text_ov.get("dimdec") is not None else 2)
        content = f"{prefix}{format_dim_value(radius, dimdec)}{suffix}"
        color = next_color()
        uid = str(el.get("uuid") or f"R_{_eid(el.get('id'))}")
        dim_scale = float(el.get("dimScale") or view.get("scale") or 1.0)
        anno = RadialAnno(
            id=uid,
            category="radial",
            color=color,
            anchor=anchor,
            path=[anchor, via, end],
            text=estimate_text(el.get("override") or {}, content, dim_scale),
            dim_scale=dim_scale,
        )
        radials.append(anno)
        legend.entries[color] = ColorLegendEntry(id=uid, category="radial")

    for el in by_ctor.get("CadChamferDimension", []):
        if el.get("pid") != view_id or el.get("inVisibleFlag"):
            continue
        # CAD mleader: attach → dogleg (horizontal shoulder) → text edge. Not a straight line.
        leaders = el.get("leaderVertexes") or []
        attach = (0.0, 0.0)
        if leaders and leaders[0]:
            attach = _v2(leaders[0][0] if isinstance(leaders[0], list) else leaders[0])
        datum = _v2(el.get("datumPt"))
        attach_s = _apply_mat(attach, el.get("matrix"))
        datum_s = _apply_mat(datum, el.get("matrix"))
        anchor = _to_view_local(attach_s, vmat)
        text_origin = _to_view_local(datum_s, vmat)
        opt = el.get("textOption") or {}
        content = str(opt.get("textContent") or "C")
        ov = opt.get("override") or el.get("override") or {}
        color = next_color()
        uid = str(el.get("uuid") or f"C_{_eid(el.get('id'))}")
        dim_scale = float(el.get("scale") or el.get("dimScale") or view.get("scale") or 1.0)
        text = estimate_text(ov if isinstance(ov, dict) else {}, content, dim_scale)
        tw = float(text.get("width") or 10.0)
        arrow_size = float(el.get("arrowSize") or 3.0)
        dogleg = float(el.get("dogleg") or 0.0)
        # CadMLeader: dl = dogleg*scale + arrow.width (arrow.width ≈ arrowSize*scale)
        dl = dogleg * dim_scale + arrow_size * dim_scale
        mid_x = text_origin.x + tw / 2.0
        b_left = anchor.x < mid_x
        text_edge = Point(x=text_origin.x if b_left else text_origin.x + tw, y=text_origin.y)
        dog_pt = Point(x=text_edge.x + ((-dl) if b_left else dl), y=text_origin.y)
        anno = RadialAnno(
            id=uid,
            category="chamfer",
            color=color,
            anchor=anchor,
            path=[anchor, dog_pt, text_edge],
            text=text,
            allow_jog=True,
            dim_scale=dim_scale,
        )
        radials.append(anno)
        legend.entries[color] = ColorLegendEntry(id=uid, category="chamfer")

    roughness: List[RoughnessAnno] = []
    for el in by_ctor.get("CadUserBlock", []):
        if el.get("pid") != view_id:
            continue
        rg = convert_roughness_block(el, vmat)
        if rg is not None:
            roughness.append(rg)

    if not linears and not diameters and not radials:
        return None

    # pad bbox with annotation extents (include full text boxes, not just anchors)
    xs = [vmin[0], vmax[0]]
    ys = [vmin[1], vmax[1]]

    def _expand_text(cx: float, cy: float, tw: float, th: float) -> None:
        xs.extend([cx - tw / 2, cx + tw / 2])
        ys.extend([cy - th / 2, cy + th / 2])

    for a in linears:
        xs += [a.p1.x, a.p2.x, a.point.x]
        ys += [a.p1.y, a.p2.y, a.point.y]
        tw = float(a.text.get("width") or 6.0)
        th = float(a.text.get("height") or 3.5)
        if a.text_point is not None:
            _expand_text(a.text_point.x, a.text_point.y, tw, th)
        else:
            _expand_text(0.5 * (a.p1.x + a.p2.x), a.point.y, tw, th)
    for a in diameters + radials:
        for p in a.path:
            xs.append(p.x)
            ys.append(p.y)
        tw = float(a.text.get("width") or 10.0)
        th = float(a.text.get("height") or 5.0)
        if a.path:
            # Chamfer: path = [attach, dogleg, textEdge]; text spans away from dogleg
            if a.category == "chamfer" and len(a.path) >= 3:
                dog, edge = a.path[-2], a.path[-1]
                direction = 1.0 if edge.x >= dog.x else -1.0
                cx = edge.x + direction * tw / 2.0
                cy = edge.y
            else:
                end = a.path[-1]
                cx, cy = end.x, end.y
            _expand_text(cx, cy, tw, th)
    for rg in roughness:
        rx0, ry0, rx1, ry1 = roughness_extent(rg)
        xs.extend([rx0, rx1])
        ys.extend([ry0, ry1])

    pad = 15.0
    bbox_min = Point(x=min(xs) - pad, y=min(ys) - pad)
    bbox_max = Point(x=max(xs) + pad, y=max(ys) + pad)

    obstacles = view_obstacles(view_id, by_ctor)
    # add view loop
    loop = view.get("loop") or []
    if len(loop) >= 2:
        obstacles.insert(
            0,
            Obstacle(
                id=f"loop_{view.get('uuid')}",
                kind="polyline",
                raw_type="view_loop",
                polyline=[Point(x=float(p[0]), y=float(p[1])) for p in loop],
                fill_role="loop",
            ),
        )

    raw_uuid = str(view.get("uuid") or view_id or "unknown")
    # Windows-safe id (uuid may contain * ? etc.)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw_uuid)
    sid = f"view_{safe}"
    tenant = str(solution_meta.get("_tenant_id") or "0")
    drawing_host = str(solution_meta.get("_drawing_host") or "https://drawing.designorder.cn")
    state = LayoutState(
        state_id=sid,
        created_at=datetime.now(timezone.utc).isoformat(),
        source={
            "drawing_id": str(solution_meta.get("name") or solution_meta.get("id")),
            "solution_id": str(solution_meta.get("id")),
            "solution_name": str(solution_meta.get("name") or ""),
            "part_number": str(solution_meta.get("_part_number") or ""),
            "tenant_id": tenant,
            "view_id": str(view_id),
            "view_uuid": raw_uuid,
            "view_type": str(view.get("viewType") or ""),
            "origin": "production_oss_doc",
            "drawing_url": f"{drawing_host}/?solutionId={solution_meta.get('id')}&tenantId={tenant}",
        },
        meta={
            "text_height": 5.0,
            "revision": 0,
            "phase_tag": "production",
            "linear_count": len(linears),
            "diameter_count": len(diameters),
            "radial_count": len(radials),
            "roughness_count": len(roughness),
            "obstacle_count": len(obstacles),
        },
        view={"bbox": {"min": bbox_min.model_dump(), "max": bbox_max.model_dump()}},
        raster=RasterMeta(
            instance_image="placeholder.png",
            width=1,
            height=1,
            origin=Point(x=0, y=0),
            pixel_size=0.25,
        ),
        color_legend=legend,
        obstacles=obstacles,
        linears=linears,
        diameters=diameters,
        radials=radials,
        roughness=roughness,
        linear_groups=[
            {
                "group_id": "default_linear",
                "member_ids": [x.id for x in linears],
            }
        ],
    )
    return state


def _node_id(node: Dict[str, Any]) -> Optional[int]:
    return _eid(node.get("id"))


def resolve_part_view_group_ids(by_ctor: Dict[str, List[Dict[str, Any]]], part: str) -> Tuple[set, Dict[str, Any]]:
    """CadTreeNode(name) → CadViewGroup → CadView.pid(viewGroupId).

    Returns (view_group_ids, part_meta).
    """
    part = (part or "").strip()
    nodes = []
    for n in by_ctor.get("CadTreeNode", []):
        name = str(n.get("name") or "")
        pure = str(n.get("pureName") or "")
        if name == part or pure == part:
            nodes.append(n)
    if not nodes:
        raise RuntimeError(f"CadTreeNode not found for part={part!r}")
    node_ids = {_node_id(n) for n in nodes}
    node_ids.discard(None)
    vg_ids: set = set()
    for g in by_ctor.get("CadViewGroup", []):
        tid = g.get("treeNodeId")
        ds = g.get("dataSourceTreeNodeIds") or []
        if tid in node_ids or any(x in node_ids for x in ds):
            gid = _eid(g.get("id"))
            if gid is not None:
                vg_ids.add(gid)
    if not vg_ids:
        raise RuntimeError(f"CadViewGroup not found for part={part!r} node_ids={sorted(node_ids)}")
    meta = {
        "part_number": part,
        "tree_node_ids": sorted(node_ids),
        "view_group_ids": sorted(vg_ids),
        "tree_node_name": str(nodes[0].get("name") or part),
    }
    return vg_ids, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch production drawing → Layout Agent dataset")
    ap.add_argument("--solution-id", default=DEFAULT_SOLUTION_ID)
    ap.add_argument("--tenant-id", default=DEFAULT_TENANT_ID)
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--oss-prefix", default=DEFAULT_OSS)
    ap.add_argument("--token", default=None, help="x-access-token; env LAYOUT_AGENT_TOKEN or built-in service token")
    ap.add_argument("--max-views", type=int, default=12, help="max views to export (by dim count)")
    ap.add_argument(
        "--view-types",
        default="front,left,top",
        help="comma-separated viewType filter; empty = all types",
    )
    ap.add_argument(
        "--part",
        default="",
        help="filter by part number(s): CadTreeNode.name/pureName; comma-separated for multiple",
    )
    ap.add_argument("--out", default="data/datasets/prod_070-3579")
    ap.add_argument("--reuse-raw", action="store_true", help="reuse already downloaded OSS JSON")
    ap.add_argument(
        "--local-doc",
        default="",
        help="use a local solution JSON (or .json.gz) instead of downloading OSS — "
        "for post-布局 exports when the scheme is read-only / not yet saved",
    )
    ap.add_argument(
        "--merge-into",
        default="",
        help="merge converted views into an existing dataset directory (updates its manifest.json)",
    )
    ap.add_argument(
        "--part-suffix",
        default="",
        help="append to part_number in manifest (e.g. ·布局后)",
    )
    ap.add_argument(
        "--id-suffix",
        default="",
        help="append to view state_id / snapshot filenames (e.g. _layout_after) to avoid clobbering",
    )
    ap.add_argument(
        "--phase",
        default="",
        help="meta.phase_tag + manifest item phase (e.g. layout_after)",
    )
    ap.add_argument(
        "--drawing-host",
        default="https://drawing.designorder.cn",
        help="frontend drawing URL host for source.drawing_url",
    )
    args = ap.parse_args()
    if not args.token:
        import os

        args.token = os.getenv("LAYOUT_AGENT_TOKEN") or DEFAULT_TOKEN

    cfg = load_config()
    out_root = Path(args.merge_into or args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    raw_dir = ROOT / "data" / "raw" / f"solution_{args.solution_id}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    meta_path = raw_dir / "solution_meta.json"
    doc_path = raw_dir / "solution_doc.json"

    if args.local_doc:
        local = Path(args.local_doc)
        if not local.is_absolute():
            local = (Path.cwd() / local).resolve()
        if not local.exists():
            raise FileNotFoundError(f"--local-doc not found: {local}")
        print(f"loading local doc: {local}")
        doc = load_json_file(local)
        # keep a copy in raw cache for audit
        shutil.copy2(local, raw_dir / f"solution_doc.local{local.suffix}")
        if meta_path.exists():
            solution_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            solution_meta = {
                "id": args.solution_id,
                "name": f"local_{args.solution_id}",
            }
            meta_path.write_text(json.dumps(solution_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.reuse_raw and doc_path.exists() and meta_path.exists():
        print(f"reusing raw cache: {raw_dir}")
        solution_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        doc = load_json_file(doc_path)
    else:
        print(f"fetching solution {args.solution_id} ...")
        solution_meta = fetch_solution_meta(args.solution_id, args.tenant_id, args.base_url, args.token)
        meta_path.write_text(json.dumps(solution_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        key = solution_meta.get("solutionSolutionWebDocument")
        if not key:
            raise RuntimeError("solutionSolutionWebDocument missing")
        print(f"downloading OSS doc: {key}")
        doc = download_oss_json(args.oss_prefix, key, doc_path)
        proj_key = solution_meta.get("projectWebDocument")
        if proj_key:
            download_oss_json(args.oss_prefix, proj_key, raw_dir / "project_doc.json")

    solution_meta["_tenant_id"] = args.tenant_id
    solution_meta["_drawing_host"] = args.drawing_host

    by_ctor = build_index(doc)
    views = by_ctor.get("CadView", [])
    want_types = {t.strip() for t in (args.view_types or "").split(",") if t.strip()}
    part_metas: List[Dict[str, Any]] = []
    vg_to_part: Dict[int, str] = {}
    allowed_vg_ids: Optional[set] = None
    part_list = [p.strip() for p in (args.part or "").split(",") if p.strip()]
    if part_list:
        allowed_vg_ids = set()
        for part in part_list:
            vg_ids, meta = resolve_part_view_group_ids(by_ctor, part)
            part_metas.append(meta)
            for gid in vg_ids:
                vg_to_part[int(gid)] = part
                allowed_vg_ids.add(gid)
            print(
                f"part filter {part}: tree_nodes={meta['tree_node_ids']} "
                f"view_groups={meta['view_group_ids']}"
            )
        if len(part_list) == 1:
            solution_meta["_part_number"] = part_list[0]

    scored: List[Tuple[int, Dict[str, Any], str]] = []
    for v in views:
        vg_pid = v.get("pid")
        if allowed_vg_ids is not None and vg_pid not in allowed_vg_ids:
            continue
        vt = str(v.get("viewType") or "")
        if want_types and vt not in want_types:
            continue
        vid = _eid(v.get("id"))
        n = 0
        for ctor in ("CadAlignedDimension", "CadDiameterDimension", "CadRadialDimension", "CadChamferDimension"):
            n += sum(1 for el in by_ctor.get(ctor, []) if el.get("pid") == vid and not el.get("inVisibleFlag"))
        if n > 0:
            part_no = vg_to_part.get(int(vg_pid), "") if vg_pid is not None else ""
            scored.append((n, v, part_no))
    scored.sort(key=lambda x: (-x[0], x[2], str(x[1].get("viewType") or "")))
    selected = scored[: args.max_views]
    print(f"selected {len(selected)} / {len(scored)} views with dimensions")

    snap = out_root / "snapshots"
    snap.mkdir(parents=True, exist_ok=True)
    # use slightly finer pixels for real drawings; cap so huge sheets don't OOM
    base_ps = float(cfg.get("perceive", {}).get("pixel_size", 0.35))
    cfg.setdefault("perceive", {})["pixel_size"] = base_ps
    perceiver = Perceiver(cfg)
    # expand palette if needed by cycling
    palette = ACTIVE_PALETTE * 20

    def _adaptive_ps(st: LayoutState, max_px: int = 2048) -> float:
        bbox = st.view.get("bbox") or {}
        bmin = bbox.get("min", {"x": 0, "y": 0})
        bmax = bbox.get("max", {"x": 100, "y": 80})
        span = max(
            float(bmax["x"]) - float(bmin["x"]),
            float(bmax["y"]) - float(bmin["y"]),
            1e-6,
        )
        return max(base_ps, span / max_px)

    manifest_items = []
    for n_dims, view, part_no in selected:
        if part_no:
            solution_meta["_part_number"] = part_no
        elif not part_list:
            solution_meta.pop("_part_number", None)
        try:
            state = convert_view(view, by_ctor, solution_meta, palette)
        except Exception as e:  # noqa: BLE001
            print(f"  skip view {view.get('uuid')}: convert failed {e}")
            continue
        if state is None:
            continue
        if args.id_suffix:
            state.state_id = f"{state.state_id}{args.id_suffix}"
        if args.phase:
            state.meta = dict(state.meta or {})
            state.meta["phase_tag"] = args.phase
        case_id = state.state_id
        rel_inst = f"snapshots/{case_id}.instance.png"
        rel_obs = f"snapshots/{case_id}.obstacle.png"
        rel_state = f"snapshots/{case_id}.state.json"
        rel_legend = f"snapshots/{case_id}.legend.json"

        perceiver.pixel_size = _adaptive_ps(state)
        try:
            state = perceiver.render_icm(state, out_root / rel_inst, out_root / rel_obs)
        except MemoryError:
            perceiver.pixel_size = _adaptive_ps(state, max_px=1024)
            try:
                state = perceiver.render_icm(state, out_root / rel_inst, out_root / rel_obs)
            except MemoryError:
                print(f"  skip view {view.get('uuid')}: ICM MemoryError")
                continue
        state.raster.instance_image = rel_inst
        state.raster.obstacle_image = rel_obs

        (out_root / rel_legend).write_text(
            json.dumps(state.color_legend.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_root / rel_state).write_text(
            json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        base_part = str(solution_meta.get("_part_number") or part_no or "")
        part_number = f"{base_part}{args.part_suffix}" if base_part else args.part_suffix
        tags = [
            "production",
            str(view.get("viewType") or "view"),
            f"L{len(state.linears)}",
            f"D{len(state.diameters)}",
            f"R{len(state.radials)}",
        ]
        if args.phase:
            tags.append(args.phase)
        if part_number:
            tags.insert(1, part_number)
        item: Dict[str, Any] = {
            "state_path": rel_state,
            "instance_image_path": rel_inst,
            "legend_path": rel_legend,
            "obstacle_image_path": rel_obs,
            "tags": tags,
            "split": "prod",
            "solution_id": str(args.solution_id),
            "part_number": part_number,
            "view_uuid": view.get("uuid"),
            "view_type": view.get("viewType"),
            "dim_count": len(state.linears) + len(state.diameters) + len(state.radials),
            "linear_count": len(state.linears),
            "diameter_count": len(state.diameters),
            "radial_count": len(state.radials),
        }
        if args.phase:
            item["phase"] = args.phase
            item["base_view"] = case_id[: -len(args.id_suffix)] if args.id_suffix and case_id.endswith(args.id_suffix) else case_id
        manifest_items.append(item)
        print(
            f"  [{part_number or '-'}] {case_id}: linear={len(state.linears)} dia={len(state.diameters)} "
            f"rad={len(state.radials)} obs={len(state.obstacles)} (raw_dims={n_dims})"
        )

    man_path = out_root / "manifest.json"
    if args.merge_into and man_path.exists():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        drop_parts = {i.get("part_number") for i in manifest_items if i.get("part_number")}
        drop_paths = {i.get("state_path") for i in manifest_items}
        kept = [
            i
            for i in manifest.get("items") or []
            if i.get("state_path") not in drop_paths
            and (not drop_parts or i.get("part_number") not in drop_parts)
        ]
        kept.extend(manifest_items)
        manifest["items"] = kept
        parts = list(manifest.get("part_numbers") or [])
        for i in manifest_items:
            pn = i.get("part_number") or ""
            if pn and pn not in parts:
                parts.append(pn)
        # also keep base part numbers from --part
        for p in part_list:
            if p and p not in parts:
                parts.append(p)
        manifest["part_numbers"] = parts
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        if args.local_doc:
            manifest["note"] = (
                (manifest.get("note") or "")
                + f" | merged local layout-after from {args.local_doc}"
            ).strip(" |")
        man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"merged -> {man_path} (+{len(manifest_items)} views, total {len(kept)})")
    else:
        ds_name = (
            f"prod_{args.solution_id}"
            if len(part_list) > 1
            else (f"prod_{part_list[0]}" if part_list else f"prod_{solution_meta.get('name') or args.solution_id}")
        )
        merged_parts = list(part_list)
        for i in manifest_items:
            pn = i.get("part_number") or ""
            if pn and pn not in merged_parts:
                merged_parts.append(pn)
        manifest = {
            "dataset_name": ds_name,
            "schema_version": "agent1.1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "owner": "layout_agent",
            "note": (
                "Local post-布局 document import"
                if args.local_doc
                else "Real production drawing from designorder OSS via mistSolution/selectById"
            ),
            "solution_id": args.solution_id,
            "tenant_id": args.tenant_id,
            "solution_name": solution_meta.get("name"),
            "part_number": merged_parts[0] if len(merged_parts) == 1 else "",
            "part_numbers": merged_parts,
            "part_meta": part_metas[0] if len(part_metas) == 1 else part_metas,
            "drawing_url": f"{args.drawing_host}/?solutionId={args.solution_id}&tenantId={args.tenant_id}",
            "platform_project_url": (
                f"https://v3.private.designorder.cn/shanshe2/project/{solution_meta.get('solutionProjectId')}"
                f"?tenantId={args.tenant_id}&app=shanshe2"
                if "private" in args.base_url
                else (
                    f"https://v3.designorder.cn/shanshe2/project/{solution_meta.get('solutionProjectId')}"
                    f"?tenantId={args.tenant_id}&app=shanshe2"
                )
            ),
            "raw_cache": str(raw_dir.as_posix()),
            "items": manifest_items,
        }
        man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"manifest -> {man_path} ({len(manifest_items)} views)")


if __name__ == "__main__":
    main()
