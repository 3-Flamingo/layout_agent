"""CAD roughness glyphs and dimension upper/lower deviations.

Geometry and strings follow drawing-2d:
  - CadDimensionUtil.overrideToTol (commom/common.ts)
  - RoughnessBlockRecord.createChildren (roughness_block_down_record.ts)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from layout_agent.models import Point, RoughnessAnno

# CadView DefaultDimStyle + RoughnessBlockRecord._calcDrawOpt
_ROUGHNESS_DRAW_OPTS = (
    (2.5, 3.5, 7.5),
    (3.5, 5.0, 10.5),
    (5.0, 7.0, 15.0),
    (7.0, 10.0, 21.0),
    (10.0, 14.0, 30.0),
    (14.0, 20.0, 42.0),
    (20.0, 28.0, 60.0),
)

_STYLE3_COUNT = {1.6: 3, 3.2: 2, 6.3: 1}


def cad_to_fixed(num: float, max_fraction_digits: int) -> str:
    """Match CadDimensionUtil.toFixed (strip trailing zeros)."""
    digits = max(0, int(max_fraction_digits))
    fixed = f"{num + 1e-7:.{digits}f}"
    if "." in fixed:
        while fixed.endswith("0"):
            fixed = fixed[:-1]
        if fixed.endswith("."):
            fixed = fixed[:-1]
    return fixed


def format_cad_tolerance(
    tolerance: Optional[Dict[str, Any]],
    dimtxt: float,
) -> Optional[Dict[str, Any]]:
    """Return drawable up/down strings. None when CAD would omit the tol.

    CadDimensionUtil.overrideToTol:
      dimtp == dimtm → single ``±abs`` in ``down``
      else stacked ``up`` / ``down`` (dimtm>0 means minus).
    """
    if not isinstance(tolerance, dict) or not tolerance.get("dimtol"):
        return None
    raw_tp = tolerance.get("dimtp")
    raw_tm = tolerance.get("dimtm")
    if not raw_tp and not raw_tm:
        return None
    precision = int(tolerance.get("precision") if tolerance.get("precision") is not None else 2)
    try:
        ratio = float(tolerance["dimtfac"]) if tolerance.get("dimtfac") is not None else (1.0 / 3.0)
    except (TypeError, ValueError):
        ratio = 1.0 / 3.0
    try:
        aligned = int(tolerance.get("dimtolj") or 0)
    except (TypeError, ValueError):
        aligned = 0
    height = max(float(dimtxt) * ratio, 0.1)

    def _parsed(raw: Any) -> Optional[float]:
        if raw is None:
            return None
        try:
            return float(cad_to_fixed(float(raw), precision))
        except (TypeError, ValueError):
            return None

    # JS treats 0 as missing after `dimtp ? ... : '0'`
    tp = _parsed(raw_tp)
    tm = _parsed(raw_tm)

    if tp is not None and tm is not None and tp == tm:
        return {
            "kind": "symmetric",
            "up": "",
            "down": f"±{abs(tp)}",
            "height": height,
            "aligned": aligned,
        }

    if tp:
        up = f"+{abs(tp)}" if tp > 0 else f"-{abs(tp)}"
    else:
        up = "0"
    if tm:
        down = f"+{abs(tm)}" if tm < 0 else f"-{abs(tm)}"
    else:
        down = "0"
    return {
        "kind": "deviation",
        "up": up,
        "down": down,
        "height": height,
        "aligned": aligned,
    }


def attach_tolerance(text: Dict[str, Any], override: Optional[Dict[str, Any]], dim_scale: float) -> Dict[str, Any]:
    """Widen a dim text dict with CAD tolerance and return it (mutates a copy)."""
    out = dict(text or {})
    dimtxt = float(out.get("dimtxt") or 3.5)
    ov = override if isinstance(override, dict) else {}
    tol = format_cad_tolerance(ov.get("tolerance") if isinstance(ov.get("tolerance"), dict) else None, dimtxt)
    if not tol:
        out.pop("tolerance", None)
        return out
    scale = max(float(dim_scale or 1.0), 1e-6)
    tol_h = float(tol["height"]) * scale
    extra = max(len(tol.get("up") or ""), len(tol.get("down") or ""), 1) * tol_h * 0.62
    try:
        width = float(out.get("width") or 0.0)
    except (TypeError, ValueError):
        width = 0.0
    out["width"] = width + extra + 0.35 * scale
    if tol.get("kind") == "deviation" and (tol.get("up") and tol.get("down")):
        try:
            height = float(out.get("height") or dimtxt * scale)
        except (TypeError, ValueError):
            height = dimtxt * scale
        out["height"] = max(height, 2.15 * tol_h)
    out["tolerance"] = {
        "kind": tol["kind"],
        "up": tol["up"],
        "down": tol["down"],
        "height": float(tol["height"]),
        "aligned": int(tol["aligned"]),
    }
    return out


@dataclass
class RoughnessGeom:
    polylines: List[List[Tuple[float, float]]] = field(default_factory=list)
    circles: List[Tuple[float, float, float]] = field(default_factory=list)
    texts: List[Tuple[float, float, str, float, float]] = field(default_factory=list)


def _rot(x: float, y: float, deg: float) -> Tuple[float, float]:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return x * c - y * s, x * s + y * c


def _draw_opt(text_height: float) -> Tuple[float, float, float]:
    return min(_ROUGHNESS_DRAW_OPTS, key=lambda t: abs(t[0] - float(text_height)))


def _style3_count(value: str) -> int:
    try:
        return int(_STYLE3_COUNT.get(float(value), 1))
    except (TypeError, ValueError):
        return 1


def roughness_local_geom(
    style: int,
    value: str,
    text_height: float,
    roughness_scale: float = 1.0,
    triangle_side: float = 2.89,
    prefix: str = "Rz",
    rotation: float = 0.0,
) -> RoughnessGeom:
    """Local mm, tip at origin, +Y up. Rotation only affects which text copy CAD picks."""
    style = int(style or 1)
    scale = max(float(roughness_scale or 1.0), 1e-6)
    th, h1, h2 = _draw_opt(text_height)
    tan30 = math.tan(math.radians(30.0))
    w1 = tan30 * h1 * scale
    w2 = tan30 * h2 * scale
    flip_text = 90 < round(float(rotation)) <= 270
    text_rot = 180.0 if flip_text else 0.0
    geom = RoughnessGeom()

    def add_iso_check(closed_bar: bool, all_around: bool) -> None:
        if all_around:
            pts = [(-w2 - w1, -h2), (-w2, -h2), (0.0, 0.0), (w1, -h1)]
            if closed_bar:
                pts.append((-w1, -h1))
        else:
            pts = [(-w2, -h2), (0.0, 0.0), (w1, -h1), (-w1, -h1)]
        geom.polylines.append(pts)
        if value:
            geom.texts.append((0.0, -h1 - (h2 - h1) / 2.0, value, text_rot, th))

    if style == 1:
        add_iso_check(closed_bar=True, all_around=False)
    elif style in (2, 6):
        add_iso_check(closed_bar=(style == 2), all_around=True)
    elif style == 3:
        half = (float(triangle_side) if triangle_side else 2.89) / 2.0
        height = half / tan30
        rh, rw = height * scale, half * scale
        tri = [(-rw, -rh), (0.0, 0.0), (rw, -rh), (-rw, -rh)]
        count = _style3_count(value)
        span = 2.0 * rw
        for i in range(count):
            dx = i * span - (count - 1) * rw
            geom.polylines.append([(x + dx, y) for x, y in tri])
    elif style == 4:
        geom.polylines.append([(-w2, -h2), (0.0, 0.0), (w1, -h1)])
        cy = -h1 * (2.33 / 3.5) * scale
        cr = h1 * (2.33 / 2.0 / 3.5) * scale
        geom.circles.append((0.0, cy, cr))
        if value:
            geom.texts.append((0.0, cy - cr - th * 0.85, value, text_rot, th))
    elif style == 7:
        height, tri_h, text_h = 7.5, 3.5, 2.5
        width = height * tan30
        tri_w = tan30 * tri_h
        code = f"{prefix or 'Rz'}{value}"
        dogleg = (tri_h - text_h) / 2.0
        tail = tri_w * max(len(code), 1) + dogleg
        end = (-width - tail, -height)
        geom.polylines.append(
            [(-tri_w, -tri_h), (tri_w, -tri_h), (0.0, 0.0), (-width, -height), end]
        )
        if code:
            tx = (end[0] + (-width)) / 2.0
            ty = (end[1] + (-height)) / 2.0 + text_h * 0.25
            geom.texts.append((tx, ty, code, 180.0, text_h))
    else:
        add_iso_check(closed_bar=True, all_around=False)
    return geom


def roughness_world_geom(anno: RoughnessAnno) -> RoughnessGeom:
    local = roughness_local_geom(
        style=int(anno.style or 1),
        value=str(anno.value or ""),
        text_height=float(anno.text_height or 2.5),
        roughness_scale=float(anno.roughness_scale or 1.0),
        triangle_side=float(anno.triangle_side or 2.89),
        prefix=str(anno.prefix or "Rz"),
        rotation=float(anno.rotation or 0.0),
    )
    ox, oy = float(anno.insert.x), float(anno.insert.y)
    rot = float(anno.rotation or 0.0)
    sc = max(float(anno.scale or 1.0), 1e-6)

    def xf(x: float, y: float) -> Tuple[float, float]:
        rx, ry = _rot(x * sc, y * sc, rot)
        return rx + ox, ry + oy

    world = RoughnessGeom()
    for poly in local.polylines:
        world.polylines.append([xf(x, y) for x, y in poly])
    for cx, cy, r in local.circles:
        wx, wy = xf(cx, cy)
        world.circles.append((wx, wy, r * sc))
    for tx, ty, content, tdeg, th in local.texts:
        wx, wy = xf(tx, ty)
        world.texts.append((wx, wy, content, tdeg + rot, th * sc))
    return world


def roughness_extent(anno: RoughnessAnno, pad: float = 2.0) -> Tuple[float, float, float, float]:
    geom = roughness_world_geom(anno)
    xs: List[float] = [anno.insert.x]
    ys: List[float] = [anno.insert.y]
    for poly in geom.polylines:
        for x, y in poly:
            xs.append(x)
            ys.append(y)
    for cx, cy, r in geom.circles:
        xs.extend((cx - r, cx + r))
        ys.extend((cy - r, cy + r))
    for tx, ty, _c, _r, th in geom.texts:
        xs.extend((tx - th * 2, tx + th * 2))
        ys.extend((ty - th, ty + th))
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def roughness_value_from_block(el: Dict[str, Any]) -> str:
    args = el.get("args") if isinstance(el.get("args"), dict) else {}
    fallback = ""
    for tag in el.get("tags") or []:
        if not isinstance(tag, dict):
            continue
        name = str(tag.get("tag") or "")
        val = str(tag.get("value") or "").strip()
        if not val:
            continue
        if "粗糙" in name or name.lower() in ("ra", "rz", "roughness"):
            return val
        if not fallback:
            fallback = val
    return fallback or str(args.get("roughnessValue") or args.get("roughnessNCValue") or "").strip()


def is_roughness_block(el: Dict[str, Any]) -> bool:
    name = str(el.get("name") or "")
    return name == "roughness_down" or name.startswith("roughness_down")
