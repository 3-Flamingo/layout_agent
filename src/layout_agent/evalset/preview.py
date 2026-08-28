"""Human-readable preview render of a LayoutState (for annotation UIs).

Unlike the ICM (unique flat colors for pixel-conflict checks), this render is
meant for people: obstacle lines in gray, dims colored by category, real text
content drawn inside the text box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from layout_agent.geometry_parts import collect_parts
from layout_agent.models import LayoutState

_BG = (24, 26, 30)
_OBSTACLE = (200, 200, 200)
_FRAME = (70, 74, 82)
_CATEGORY_COLOR = {
    "linear": (255, 214, 68),
    "diameter": (86, 208, 255),
    "radial": (255, 121, 198),
    "chamfer": (189, 147, 249),
}


def _font(px: int) -> ImageFont.FreeTypeFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def render_preview(
    state: LayoutState,
    out_path: Path,
    max_px: int = 1400,
    pad_mm: float = 10.0,
) -> Path:
    bbox = state.view.get("bbox") or {}
    bmin = bbox.get("min", {"x": 0, "y": 0})
    bmax = bbox.get("max", {"x": 100, "y": 80})
    xmin, ymin = float(bmin["x"]) - pad_mm, float(bmin["y"]) - pad_mm
    xmax, ymax = float(bmax["x"]) + pad_mm, float(bmax["y"]) + pad_mm
    w_mm = max(xmax - xmin, 1e-6)
    h_mm = max(ymax - ymin, 1e-6)
    scale = max_px / max(w_mm, h_mm)
    width = max(64, int(w_mm * scale))
    height = max(64, int(h_mm * scale))

    def px(x: float, y: float) -> Tuple[float, float]:
        return (x - xmin) * scale, (ymax - y) * scale  # y_up world -> y_down image

    img = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(img)

    # view frame
    fx0, fy0 = px(float(bmin["x"]), float(bmax["y"]))
    fx1, fy1 = px(float(bmax["x"]), float(bmin["y"]))
    draw.rectangle([fx0, fy0, fx1, fy1], outline=_FRAME, width=1)

    # obstacles
    for ob in state.obstacles:
        pts = ob.polyline or []
        if len(pts) >= 2:
            xy = [px(p.x, p.y) for p in pts]
            if ob.fill_role == "loop" and len(pts) >= 3:
                xy.append(xy[0])
            draw.line(xy, fill=_OBSTACLE, width=1)
        elif ob.aabb is not None:
            a = px(ob.aabb.min.x, ob.aabb.max.y)
            b = px(ob.aabb.max.x, ob.aabb.min.y)
            draw.rectangle([a, b], outline=_OBSTACLE, width=1)

    # annotations via the same decomposition the evaluator uses
    parts = collect_parts(state)
    contents = {}
    for lin in state.linears:
        contents[lin.id] = str((lin.text or {}).get("content") or _fmt(lin.value))
    for a in list(state.diameters) + list(state.radials):
        contents[a.id] = str((a.text or {}).get("content") or "")

    for aid, p in parts.items():
        color = _CATEGORY_COLOR.get(p.category, (255, 255, 255))
        for seg in p.extensions:
            draw.line([px(seg.a.x, seg.a.y), px(seg.b.x, seg.b.y)], fill=_dim(color), width=1)
        for seg in p.dim_lines + p.leaders:
            draw.line([px(seg.a.x, seg.a.y), px(seg.b.x, seg.b.y)], fill=color, width=1)
        # text box + content
        t = p.text
        x0, y0 = px(t.xmin, t.ymax)
        x1, y1 = px(t.xmax, t.ymin)
        box_h = max(6, int(y1 - y0))
        font = _font(max(8, int(box_h * 0.85)))
        txt = contents.get(aid, "")
        if txt:
            draw.text(((x0 + x1) / 2, (y0 + y1) / 2), txt, fill=color, font=font, anchor="mm")
        else:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.10g}"


def _dim(c: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(int(x * 0.55) for x in c)  # dimmed variant for extension lines
