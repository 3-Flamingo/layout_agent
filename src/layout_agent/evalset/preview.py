"""Render a paper CAD sheet from a production LayoutState.

Geometry comes from the dumped CadView (real 投图轮廓). Dimensions are drawn
at the state's current poses, so a perturbation that moves text / dim lines /
leaders is visible on the image. Perturbed layouts do not exist on the website,
so both T1 and T2 sides are produced this way.

Text placement follows CadView dimtad=UP: numbers sit above the dimension
line (left of a vertical line) with dimgap, and the line stays continuous.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from layout_agent.cad_marks import roughness_extent, roughness_world_geom
from layout_agent.geometry_parts import AABB, AnnoParts, Point, Segment, collect_parts
from layout_agent.models import LayoutState, LinearAnno, RadialAnno, RoughnessAnno

# Paper CAD (投图) palette — white sheet, dark geometry, CAD-like dim colors.
_BG = (255, 255, 255)
_LOOP_FILL = (248, 248, 248)
_OBSTACLE = (40, 40, 40)
_FRAME = (0, 150, 70)
_CATEGORY_COLOR = {
    "linear": (0, 128, 0),
    "diameter": (0, 90, 180),
    "radial": (160, 0, 140),
    "chamfer": (120, 70, 0),
}
# Official CAD color index 3 (green) for roughness_down.
_ROUGHNESS_COLOR = (0, 176, 80)

# CadView DefaultDimStyle.text.dimgap / dimtxt (mm at dimScale=1).
_DIMGAP_MM = 0.625
_DIMTXT_MM = 3.5
_ARROW_HALF = 0.35  # matches geometry_parts V-wing half-width / arrow_width

_FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
# Production uses romans.shx + gbcbig.shx. Closest common TTFs:
# Arial Narrow ≈ ISOCP proportions; SimSun/STSong for CJK.
_LATIN_FONTS = (
    "ARIALN.TTF",
    "arialn.ttf",
    "cour.ttf",
    "consola.ttf",
    "arial.ttf",
)
_CJK_FONTS = (
    "STSONG.TTF",
    "simsun.ttc",
    "simhei.ttf",
    "msyh.ttc",
    "msyh.ttf",
)

_FONT_CACHE: dict = {}


def _has_cjk(txt: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in txt)


def _truetype(name: str, px: int) -> Optional[ImageFont.FreeTypeFont]:
    px = max(6, int(px))
    key = (name, px)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = [name, str(_FONT_DIR / name)]
    for path in candidates:
        try:
            font = ImageFont.truetype(path, px)
            _FONT_CACHE[key] = font
            return font
        except OSError:
            continue
    return None


def _font_for_height(txt: str, height_px: float) -> ImageFont.ImageFont:
    """TTF whose glyph bbox height matches CAD dimtxt in pixels."""
    height_px = max(6.0, float(height_px))
    names = _CJK_FONTS + _LATIN_FONTS if _has_cjk(txt) else _LATIN_FONTS + _CJK_FONTS
    sample = txt if txt.strip() else "0"
    for name in names:
        font = _truetype(name, int(round(height_px)))
        if font is None:
            continue
        bb = font.getbbox(sample)
        h = max((bb[3] - bb[1]), 1)
        scaled = max(6, int(round(height_px * height_px / h)))
        fitted = _truetype(name, scaled) or font
        _FONT_CACHE[("fit", name, round(height_px, 2), sample)] = fitted
        return fitted
    return ImageFont.load_default()


def _dim_scale(obj) -> float:
    if obj is None:
        return 1.0
    text = getattr(obj, "text", None) or {}
    for raw in (getattr(obj, "dim_scale", None), text.get("dim_scale") if isinstance(text, dict) else None):
        if raw is not None:
            try:
                v = float(raw)
                if v > 1e-9:
                    return v
            except (TypeError, ValueError):
                pass
    return 1.0


def _text_h_mm(obj) -> float:
    text = getattr(obj, "text", None) or {}
    dimtxt = text.get("dimtxt") if isinstance(text, dict) else None
    try:
        h = float(dimtxt) if dimtxt is not None else _DIMTXT_MM
    except (TypeError, ValueError):
        h = _DIMTXT_MM
    return h * _dim_scale(obj)


def _cad_up(ux: float, uy: float) -> Tuple[float, float]:
    """CadView DimCommonUtil.adjustBaseDir: perpendicular 'up/left' of a dim line."""
    nx, ny = -uy, ux
    nlen = math.hypot(nx, ny) or 1.0
    nx, ny = nx / nlen, ny / nlen
    if (abs(ny) < 1e-3 and nx > 0) or ny < -1e-3:
        nx, ny = -nx, -ny
    return nx, ny


def _readable_deg(dim_angle: Optional[float]) -> Optional[float]:
    if dim_angle is None:
        return None
    deg = float(dim_angle)
    while deg > 90:
        deg -= 180
    while deg <= -90:
        deg += 180
    return deg


def _near(a: Point, b: Point, eps: float) -> bool:
    return math.hypot(a.x - b.x, a.y - b.y) <= eps


def _seg_len(seg: Segment) -> float:
    return math.hypot(seg.b.x - seg.a.x, seg.b.y - seg.a.y)


def _is_arrow_wing(seg: Segment, feet: Sequence[Point], aw: float) -> bool:
    expected = math.hypot(aw, aw * _ARROW_HALF)
    if abs(_seg_len(seg) - expected) > max(0.45, 0.3 * aw):
        return False
    tol = max(0.08, 0.05 * aw)
    return any(_near(seg.a, f, tol) or _near(seg.b, f, tol) for f in feet)


def _perp_dist_to_infinite_line(p: Point, a: Point, b: Point) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return math.hypot(p.x - a.x, p.y - a.y)
    return abs((p.x - a.x) * dy - (p.y - a.y) * dx) / L


def _text_on_dim_line(text: AABB, dim_lines: Sequence[Segment], th_mm: float) -> bool:
    if not dim_lines:
        return False
    c = text.center
    # Use the infinite line so a gap around the text still counts as "on the dim line".
    return any(
        _perp_dist_to_infinite_line(c, seg.a, seg.b) < max(0.45 * th_mm, 0.8)
        for seg in dim_lines
    )


def _arrow_poly(tip: Point, along: Tuple[float, float], aw: float) -> Sequence[Point]:
    """Filled closed arrow: tip on the extension, body along `along` (unit)."""
    ux, uy = along
    nx, ny = -uy * aw * _ARROW_HALF, ux * aw * _ARROW_HALF
    base = Point(x=tip.x + ux * aw, y=tip.y + uy * aw)
    return (
        tip,
        Point(x=base.x + nx, y=base.y + ny),
        Point(x=base.x - nx, y=base.y - ny),
    )


def render_preview(
    state: LayoutState,
    out_path: Path,
    max_px: int = 1600,
    pad_mm: float = 8.0,
) -> Path:
    bbox = state.view.get("bbox") or {}
    bmin = bbox.get("min", {"x": 0, "y": 0})
    bmax = bbox.get("max", {"x": 100, "y": 80})
    xmin, ymin = float(bmin["x"]) - pad_mm, float(bmin["y"]) - pad_mm
    xmax, ymax = float(bmax["x"]) + pad_mm, float(bmax["y"]) + pad_mm
    # Perturbed text/dim-lines can sit outside the original view bbox.
    parts = collect_parts(state)
    for p in parts.values():
        fp = p.footprint()
        xmin = min(xmin, fp.xmin - pad_mm)
        ymin = min(ymin, fp.ymin - pad_mm)
        xmax = max(xmax, fp.xmax + pad_mm)
        ymax = max(ymax, fp.ymax + pad_mm)
    for rg in getattr(state, "roughness", None) or []:
        rx0, ry0, rx1, ry1 = roughness_extent(rg)
        xmin = min(xmin, rx0 - pad_mm)
        ymin = min(ymin, ry0 - pad_mm)
        xmax = max(xmax, rx1 + pad_mm)
        ymax = max(ymax, ry1 + pad_mm)
    w_mm = max(xmax - xmin, 1e-6)
    h_mm = max(ymax - ymin, 1e-6)
    ss = 2  # supersample so filled arrows / CAD text downsample cleanly
    scale = (max_px * ss) / max(w_mm, h_mm)
    width = max(64, int(w_mm * scale))
    height = max(64, int(h_mm * scale))
    geom_w = max(1, int(round(scale * 0.22)))
    dim_w = max(1, int(round(scale * 0.16)))
    ext_w = max(1, dim_w)

    def px(x: float, y: float) -> Tuple[float, float]:
        return (x - xmin) * scale, (ymax - y) * scale  # y_up world -> y_down image

    img = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(img)

    # view frame (green, like production drawing sheets)
    fx0, fy0 = px(float(bmin["x"]), float(bmax["y"]))
    fx1, fy1 = px(float(bmax["x"]), float(bmin["y"]))
    draw.rectangle([fx0, fy0, fx1, fy1], outline=_FRAME, width=max(1, geom_w))

    # part loops first (fill), then all obstacle polylines
    for ob in state.obstacles:
        pts = ob.polyline or []
        if len(pts) >= 3 and ob.fill_role == "loop":
            xy = [px(p.x, p.y) for p in pts]
            draw.polygon(xy, fill=_LOOP_FILL, outline=_OBSTACLE)
    for ob in state.obstacles:
        pts = ob.polyline or []
        if len(pts) >= 2:
            xy = [px(p.x, p.y) for p in pts]
            if ob.fill_role == "loop" and len(pts) >= 3:
                xy.append(xy[0])
            draw.line(xy, fill=_OBSTACLE, width=geom_w)
        elif ob.aabb is not None:
            a = px(ob.aabb.min.x, ob.aabb.max.y)
            b = px(ob.aabb.max.x, ob.aabb.min.y)
            draw.rectangle([a, b], outline=_OBSTACLE, width=geom_w)

    contents = {}
    lin_by_id = {lin.id: lin for lin in state.linears}
    rad_by_id = {a.id: a for a in list(state.diameters) + list(state.radials)}
    for lin in state.linears:
        contents[lin.id] = str((lin.text or {}).get("content") or _fmt(lin.value))
    for a in list(state.diameters) + list(state.radials):
        contents[a.id] = str((a.text or {}).get("content") or "")

    for aid, p in parts.items():
        color = _CATEGORY_COLOR.get(p.category, (20, 20, 20))
        ext_color = _mix(color, (80, 80, 80), 0.45)
        lin = lin_by_id.get(aid)
        rad = rad_by_id.get(aid)
        src = lin if lin is not None else rad
        th_mm = _text_h_mm(src) if src is not None else max(p.text.ymax - p.text.ymin, _DIMTXT_MM)
        gap_mm = _DIMGAP_MM * _dim_scale(src)
        aw = float(getattr(lin, "arrow_width", None) or 2.5) if lin is not None else 2.5

        for seg in p.extensions:
            draw.line([px(seg.a.x, seg.a.y), px(seg.b.x, seg.b.y)], fill=ext_color, width=ext_w)

        if p.category == "linear":
            for seg in p.dim_lines:
                draw.line([px(seg.a.x, seg.a.y), px(seg.b.x, seg.b.y)], fill=color, width=dim_w)
            feet: Sequence[Point] = ()
            if p.dim_lines:
                feet = (p.dim_lines[0].a, p.dim_lines[0].b)
                _draw_linear_arrows(draw, px, color, p.dim_lines[0], aw)
            for seg in p.leaders:
                if feet and _is_arrow_wing(seg, feet, aw):
                    continue
                draw.line([px(seg.a.x, seg.a.y), px(seg.b.x, seg.b.y)], fill=color, width=dim_w)
        else:
            for seg in p.leaders:
                draw.line([px(seg.a.x, seg.a.y), px(seg.b.x, seg.b.y)], fill=color, width=dim_w)
            _draw_radial_arrow(draw, px, color, rad, aw)
            _draw_radial_landing(draw, px, color, dim_w, p, rad, th_mm)

        t = p.text
        txt = contents.get(aid, "")
        font = _font_for_height(txt, th_mm * scale)
        # 8px gap in the downsampled PNG so "above the line" stays visible at fit-zoom.
        min_gap_mm = (8.0 * ss) / max(scale, 1e-6)
        wx, wy = _text_center_mm(p, lin, rad, th_mm, gap_mm, min_gap_mm=min_gap_mm)
        cx, cy = px(wx, wy)
        if txt:
            _draw_text(draw, img, cx, cy, txt, color, font, lin, src, scale)
        else:
            x0, y0 = px(t.xmin, t.ymax)
            x1, y1 = px(t.xmax, t.ymin)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=1)

    for rg in getattr(state, "roughness", None) or []:
        _draw_roughness(draw, img, px, scale, dim_w, rg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if ss > 1:
        img = img.resize((max(64, width // ss), max(64, height // ss)), Image.Resampling.LANCZOS)
    img.save(out_path, optimize=True)
    return out_path


def _draw_linear_arrows(draw, px, color, dim: Segment, aw: float) -> None:
    dist = _seg_len(dim)
    if dist < 1e-6:
        return
    ux, uy = (dim.b.x - dim.a.x) / dist, (dim.b.y - dim.a.y) / dist
    arrows_out = 2.0 * aw + 1.0 > dist
    if arrows_out:
        polys = (
            _arrow_poly(dim.a, (-ux, -uy), aw),
            _arrow_poly(dim.b, (ux, uy), aw),
        )
    else:
        polys = (
            _arrow_poly(dim.a, (ux, uy), aw),
            _arrow_poly(dim.b, (-ux, -uy), aw),
        )
    for poly in polys:
        draw.polygon([px(pt.x, pt.y) for pt in poly], fill=color, outline=color)


def _draw_radial_arrow(draw, px, color, rad: Optional[RadialAnno], aw: float) -> None:
    if rad is None or len(rad.path) < 2:
        return
    a, b = rad.path[0], rad.path[1]
    dx, dy = a.x - b.x, a.y - b.y
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return
    poly = _arrow_poly(a, (dx / L, dy / L), aw)
    draw.polygon([px(pt.x, pt.y) for pt in poly], fill=color, outline=color)


def _draw_radial_landing(draw, px, color, dim_w: int, p: AnnoParts, rad: Optional[RadialAnno], th_mm: float) -> None:
    """Horizontal shoulder under radial/diameter text (CadView landing)."""
    if rad is None:
        return
    end = rad.path[-1] if rad.path else rad.anchor
    half_w = max(float((rad.text or {}).get("width") or (p.text.xmax - p.text.xmin)), th_mm) / 2.0
    a = Point(x=end.x - half_w, y=end.y)
    b = Point(x=end.x + half_w, y=end.y)
    draw.line([px(a.x, a.y), px(b.x, b.y)], fill=color, width=dim_w)


def _text_center_mm(
    p: AnnoParts,
    lin: Optional[LinearAnno],
    rad: Optional[RadialAnno],
    th_mm: float,
    gap_mm: float,
    min_gap_mm: float = 0.0,
) -> Tuple[float, float]:
    """World-mm center of the glyphs after dimtad=UP offset.

    ``min_gap_mm`` keeps the gap readable after fit-to-view (CadView dimgap is
    0.625mm — only ~1px on a 1.6m sheet packed into 1600px).
    """
    c = p.text.center
    lift = th_mm * 0.5 + max(gap_mm, min_gap_mm)
    if lin is not None and _text_on_dim_line(p.text, p.dim_lines, th_mm) and p.dim_lines:
        seg = p.dim_lines[0]
        dist = _seg_len(seg)
        if dist > 1e-9:
            ux, uy = (seg.b.x - seg.a.x) / dist, (seg.b.y - seg.a.y) / dist
            nx, ny = _cad_up(ux, uy)
            return c.x + nx * lift, c.y + ny * lift
        nx, ny = _cad_up(1.0, 0.0)
        return c.x + nx * lift, c.y + ny * lift
    if rad is not None:
        # Path end is the old in-line center; sit above a horizontal landing.
        end = rad.path[-1] if rad.path else rad.anchor
        return end.x, end.y + lift
    return c.x, c.y


def _draw_text(draw, img, cx, cy, txt, color, font, lin, src=None, scale: float = 1.0) -> None:
    """CAD-like glyphs; rotate linear text to dim angle when set. Deviations sit to the right."""
    ang = 0.0
    if lin is not None and not getattr(lin, "dimtih", False) and not getattr(lin, "dimtoh", False):
        deg = _readable_deg(lin.dim_angle)
        if deg is not None:
            ang = -deg  # canvas y-down
    tol = None
    if src is not None:
        text = getattr(src, "text", None) or {}
        if isinstance(text, dict):
            tol = text.get("tolerance")
    layer, mx, my = _compose_dim_glyphs(txt, color, font, src, tol, scale)
    if abs(ang) < 1.0:
        img.paste(layer, (int(round(cx - mx)), int(round(cy - my))), layer)
        return
    rot = layer.rotate(ang, expand=True, resample=Image.Resampling.BICUBIC)
    img.paste(rot, (int(round(cx - rot.size[0] / 2)), int(round(cy - rot.size[1] / 2))), rot)


def _glyph_size(font, txt: str) -> Tuple[float, float]:
    if not txt:
        return 0.0, 0.0
    bb = font.getbbox(txt)
    return float(bb[2] - bb[0]), float(bb[3] - bb[1])


def _compose_dim_glyphs(txt, color, font, src, tol, scale: float):
    """RGBA layer whose (mx, my) is the center of the main measurement string."""
    mw, mh = _glyph_size(font, txt or "")
    up = str((tol or {}).get("up") or "")
    down = str((tol or {}).get("down") or "")
    kind = str((tol or {}).get("kind") or "")
    aligned = int((tol or {}).get("aligned") or 0)
    dim_scale = _dim_scale(src) if src is not None else 1.0
    try:
        tol_h_mm = float((tol or {}).get("height") or 0.0) * dim_scale
    except (TypeError, ValueError):
        tol_h_mm = 0.0
    tol_font = _font_for_height(up or down or "0", max(tol_h_mm * scale, 1.0)) if (up or down) else font
    tw_up, th_up = _glyph_size(tol_font, up)
    tw_dn, th_dn = _glyph_size(tol_font, down)
    stacked = kind == "deviation" and bool(up) and bool(down)
    gap = max(1.0, 0.12 * max(mh, 1.0))
    if stacked:
        stack_w = max(tw_up, tw_dn)
        stack_h = th_up + th_dn + max(th_dn / 3.0, 1.0)
    else:
        stack_w = tw_dn if down else tw_up
        stack_h = th_dn if down else th_up
    pad = 10.0
    extra_r = (gap + stack_w) if (up or down) else 0.0
    layer_w = int(math.ceil(mw + 2 * (extra_r + pad)))
    layer_h = int(math.ceil(max(mh, stack_h) + 2 * pad))
    layer_w = max(layer_w, 8)
    layer_h = max(layer_h, 8)
    layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    mx, my = layer_w / 2.0, layer_h / 2.0
    fill = (*color, 255)
    if txt:
        ld.text((mx, my), txt, fill=fill, font=font, anchor="mm")
    if up or down:
        # CAD: tol block sits to the right of the measurement box; dimtolj=BOTTOM by default.
        main_right = mx + mw / 2.0
        main_top, main_bot = my - mh / 2.0, my + mh / 2.0
        if stacked:
            block_h = stack_h
            if aligned == 2:  # TOP
                block_top = main_top
            elif aligned == 1:  # MIDDLE
                block_top = my - block_h / 2.0
            else:
                block_top = main_bot - block_h
            # CAD draws down first (lower), then up above it.
            dn_cx = main_right + gap + tw_dn / 2.0
            up_cx = main_right + gap + tw_up / 2.0
            dn_cy = block_top + block_h - th_dn / 2.0
            up_cy = block_top + th_up / 2.0
            if down:
                ld.text((dn_cx, dn_cy), down, fill=fill, font=tol_font, anchor="mm")
            if up:
                ld.text((up_cx, up_cy), up, fill=fill, font=tol_font, anchor="mm")
        else:
            label = down or up
            tw, th = (tw_dn, th_dn) if down else (tw_up, th_up)
            tx = main_right + gap + tw / 2.0
            if aligned == 2:
                ty = main_top + th / 2.0
            elif aligned == 1:
                ty = my
            else:
                ty = main_bot - th / 2.0
            ld.text((tx, ty), label, fill=fill, font=tol_font, anchor="mm")
    return layer, mx, my


def _draw_roughness(draw, img, px, scale: float, dim_w: int, rg: RoughnessAnno) -> None:
    geom = roughness_world_geom(rg)
    color = _ROUGHNESS_COLOR
    lw = max(1, dim_w)
    for poly in geom.polylines:
        if len(poly) < 2:
            continue
        xy = [px(x, y) for x, y in poly]
        draw.line(xy, fill=color, width=lw)
    for cx, cy, r in geom.circles:
        x0, y0 = px(cx - r, cy + r)
        x1, y1 = px(cx + r, cy - r)
        draw.ellipse([x0, y0, x1, y1], outline=color, width=lw)
    for tx, ty, content, deg, th_mm in geom.texts:
        if not content:
            continue
        font = _font_for_height(content, th_mm * scale)
        cx, cy = px(tx, ty)
        ang = -float(deg)  # world CCW → canvas
        # Keep glyphs readable: CAD already swapped the 180° copy when the block is inverted.
        _draw_plain_text(img, cx, cy, content, color, font, ang)


def _draw_plain_text(img, cx, cy, txt, color, font, ang: float) -> None:
    if abs(ang) < 1.0:
        ImageDraw.Draw(img).text((cx, cy), txt, fill=color, font=font, anchor="mm")
        return
    tmp = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    bbox = td.textbbox((0, 0), txt, font=font, anchor="lt")
    tw, th = bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8
    layer = Image.new("RGBA", (max(tw, 8), max(th, 8)), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    mx, my = layer.size[0] / 2, layer.size[1] / 2
    ld.text((mx, my), txt, fill=(*color, 255), font=font, anchor="mm")
    rot = layer.rotate(ang, expand=True, resample=Image.Resampling.BICUBIC)
    img.paste(rot, (int(cx - rot.size[0] / 2), int(cy - rot.size[1] / 2)), rot)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.10g}"


def _mix(c: Tuple[int, int, int], other: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(int(a * (1 - t) + b * t) for a, b in zip(c, other))
