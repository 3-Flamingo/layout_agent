"""Graded perturbation operators for building weak-supervised preference pairs.

Assumption (validated by the generation script): a random perturbation of a
laid-out view is worse than (or equal to) the original, so
(original, perturbed) forms a preference pair with prefer=original.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from layout_agent.geometry_parts import linear_parts
from layout_agent.models import LayoutState, Point

# level -> (min_shift_mm, max_shift_mm, n_ops)
LEVELS: Dict[str, Tuple[float, float, int]] = {
    "light": (2.0, 6.0, 1),
    "medium": (6.0, 20.0, 2),
    "heavy": (20.0, 60.0, 3),
}


def _rand_delta(rng: random.Random, lo: float, hi: float) -> Tuple[float, float]:
    mag = rng.uniform(lo, hi)
    ang = rng.uniform(0.0, 6.283185307)
    import math

    return mag * math.cos(ang), mag * math.sin(ang)


def _ensure_text_point(lin) -> Point:
    if lin.text_point is not None:
        return lin.text_point
    t = linear_parts(lin).text
    return Point(x=(t.xmin + t.xmax) / 2.0, y=(t.ymin + t.ymax) / 2.0)


def op_jitter_linear_text(state: LayoutState, rng: random.Random, lo: float, hi: float) -> Optional[str]:
    """Move one linear dim's text off its dim line (creates H_TT/H_TD/H_TOB/H_TIN)."""
    if not state.linears:
        return None
    lin = rng.choice(state.linears)
    tp = _ensure_text_point(lin)
    dx, dy = _rand_delta(rng, lo, hi)
    lin.text_point = Point(x=tp.x + dx, y=tp.y + dy)
    return f"jitter_linear_text:{lin.id}:{dx:.1f},{dy:.1f}"


def op_shift_dimline(state: LayoutState, rng: random.Random, lo: float, hi: float) -> Optional[str]:
    """Slide one linear dim line along its offset normal (text follows)."""
    if not state.linears:
        return None
    lin = rng.choice(state.linears)
    n = lin.dir
    ln = (n.x * n.x + n.y * n.y) ** 0.5 or 1.0
    mag = rng.uniform(lo, hi) * rng.choice([-1.0, 1.0])
    dx, dy = n.x / ln * mag, n.y / ln * mag
    lin.point = Point(x=lin.point.x + dx, y=lin.point.y + dy)
    if lin.text_point is not None:
        lin.text_point = Point(x=lin.text_point.x + dx, y=lin.text_point.y + dy)
    return f"shift_dimline:{lin.id}:{mag:.1f}"


def op_swap_layers(state: LayoutState, rng: random.Random, lo: float, hi: float) -> Optional[str]:
    """Swap the dim-line positions of two same-axis linears (layer disorder)."""
    def axis(lin):
        return "h" if abs(lin.dir.x) >= abs(lin.dir.y) else "v"

    groups: Dict[str, List] = {}
    for lin in state.linears:
        groups.setdefault(axis(lin), []).append(lin)
    cands = [g for g in groups.values() if len(g) >= 2]
    if not cands:
        return None
    g = rng.choice(cands)
    a, b = rng.sample(g, 2)
    da = Point(x=b.point.x - a.point.x, y=b.point.y - a.point.y)
    for lin, d in ((a, da), (b, Point(x=-da.x, y=-da.y))):
        lin.point = Point(x=lin.point.x + d.x, y=lin.point.y + d.y)
        if lin.text_point is not None:
            lin.text_point = Point(x=lin.text_point.x + d.x, y=lin.text_point.y + d.y)
    return f"swap_layers:{a.id}<->{b.id}"


def op_jitter_radial_text(state: LayoutState, rng: random.Random, lo: float, hi: float) -> Optional[str]:
    """Move one radial/diameter text end (leader end) randomly."""
    annos = list(state.diameters) + list(state.radials)
    annos = [a for a in annos if a.path]
    if not annos:
        return None
    a = rng.choice(annos)
    dx, dy = _rand_delta(rng, lo, hi)
    end = a.path[-1]
    a.path[-1] = Point(x=end.x + dx, y=end.y + dy)
    return f"jitter_radial_text:{a.id}:{dx:.1f},{dy:.1f}"


_OPS: List[Callable] = [
    op_jitter_linear_text,
    op_shift_dimline,
    op_swap_layers,
    op_jitter_radial_text,
]


def measurement_signature(state: LayoutState) -> Tuple[Tuple[Any, ...], ...]:
    """Identity of measured objects: p1/p2/value and radial anchors. Placement is ignored."""
    rows: List[Tuple[Any, ...]] = []
    for lin in state.linears:
        rows.append(
            (
                "linear",
                lin.id,
                round(float(lin.p1.x), 4),
                round(float(lin.p1.y), 4),
                round(float(lin.p2.x), 4),
                round(float(lin.p2.y), 4),
                round(float(lin.value), 4),
            )
        )
    for anno in list(state.diameters) + list(state.radials):
        rows.append(
            (
                anno.category,
                anno.id,
                round(float(anno.anchor.x), 4),
                round(float(anno.anchor.y), 4),
            )
        )
    return tuple(sorted(rows))


_FRAME_INSET_MM = 4.0


def _view_bbox(state: LayoutState) -> Tuple[float, float, float, float]:
    bbox = (state.view or {}).get("bbox") or {}
    bmin = bbox.get("min") or {}
    bmax = bbox.get("max") or {}
    xmin = float(bmin.get("x", 0.0))
    ymin = float(bmin.get("y", 0.0))
    xmax = float(bmax.get("x", xmin + 100.0))
    ymax = float(bmax.get("y", ymin + 80.0))
    if xmax <= xmin or ymax <= ymin:
        return xmin, ymin, xmin + 100.0, ymin + 80.0
    return xmin, ymin, xmax, ymax


def _inner_box(
    bbox: Tuple[float, float, float, float], inset: float = _FRAME_INSET_MM
) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    ix = min(inset, max(0.5, 0.08 * (xmax - xmin)))
    iy = min(inset, max(0.5, 0.08 * (ymax - ymin)))
    if xmax - xmin <= 2 * ix + 1.0:
        ix = max(0.25, (xmax - xmin) * 0.04)
    if ymax - ymin <= 2 * iy + 1.0:
        iy = max(0.25, (ymax - ymin) * 0.04)
    return xmin + ix, ymin + iy, xmax - ix, ymax - iy


def _clamp(value: float, lo: float, hi: float) -> float:
    if hi < lo:
        return 0.5 * (lo + hi)
    return min(max(value, lo), hi)


def _clip_xy(x: float, y: float, box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax = box
    return _clamp(x, xmin, xmax), _clamp(y, ymin, ymax)


def _point_in_box(
    x: float, y: float, box: Tuple[float, float, float, float], eps: float = 0.05
) -> bool:
    xmin, ymin, xmax, ymax = box
    return xmin - eps <= x <= xmax + eps and ymin - eps <= y <= ymax + eps


def placement_outside_frame(state: LayoutState, eps: float = 0.05) -> List[str]:
    """Movable pose points that sit outside the green view frame."""
    box = _view_bbox(state)
    bad: List[str] = []
    for lin in state.linears:
        if not _point_in_box(lin.point.x, lin.point.y, box, eps):
            bad.append(f"linear.point:{lin.id}")
        tp = lin.text_point
        if tp is not None and not _point_in_box(tp.x, tp.y, box, eps):
            bad.append(f"linear.text:{lin.id}")
    for anno in list(state.diameters) + list(state.radials):
        for i, pt in enumerate(anno.path or []):
            if not _point_in_box(pt.x, pt.y, box, eps):
                bad.append(f"{anno.category}.path{i}:{anno.id}")
    return bad


def _follow_text(lin, dx: float, dy: float) -> None:
    if lin.text_point is not None:
        lin.text_point = Point(x=lin.text_point.x + dx, y=lin.text_point.y + dy)
    else:
        lin.text_point = Point(x=lin.point.x, y=lin.point.y)


def _pick_offset(rng: random.Random, room: float) -> float:
    """Offset from the measured mid-point toward the frame, always < room."""
    room = max(room, 0.0)
    if room < 1.0:
        return 0.4 * room
    lo = min(3.0, 0.18 * room)
    hi = max(lo, 0.88 * room)
    pad = rng.uniform(lo, hi)
    if rng.random() < 0.18:
        pad = rng.uniform(min(1.0, 0.08 * room), max(1.0, 0.22 * room))
    return min(pad, 0.95 * room)


def _cmu_reanchor_linear(lin, bbox: Tuple[float, float, float, float], rng: random.Random) -> str:
    """Keep p1/p2; place the dim line in the margin between feature and view frame."""
    xmin, ymin, xmax, ymax = _inner_box(bbox)
    span_x = abs(lin.p2.x - lin.p1.x)
    span_y = abs(lin.p2.y - lin.p1.y)
    mx = _clamp(0.5 * (lin.p1.x + lin.p2.x), xmin, xmax)
    my = _clamp(0.5 * (lin.p1.y + lin.p2.y), ymin, ymax)
    if span_x >= span_y:
        above, below = ymax - my, my - ymin
        sides = [s for s, room in ((1.0, above), (-1.0, below)) if room >= 3.0]
        if not sides:
            sides = [1.0 if above >= below else -1.0]
        side = rng.choice(sides)
        ny = _clamp(my + side * _pick_offset(rng, above if side > 0 else below), ymin, ymax)
        nx = mx
    else:
        right, left = xmax - mx, mx - xmin
        sides = [s for s, room in ((1.0, right), (-1.0, left)) if room >= 3.0]
        if not sides:
            sides = [1.0 if right >= left else -1.0]
        side = rng.choice(sides)
        nx = _clamp(mx + side * _pick_offset(rng, right if side > 0 else left), xmin, xmax)
        ny = my
    dx, dy = nx - lin.point.x, ny - lin.point.y
    lin.point = Point(x=nx, y=ny)
    _follow_text(lin, dx, dy)
    vx, vy = nx - mx, ny - my
    nlen = (vx * vx + vy * vy) ** 0.5 or 1.0
    lin.dir = Point(x=vx / nlen, y=vy / nlen)
    return f"cmu_reanchor:{lin.id}:{nx:.1f},{ny:.1f}"


def _ray_limit(ax: float, ay: float, dx: float, dy: float, box: Tuple[float, float, float, float]) -> float:
    xmin, ymin, xmax, ymax = box
    hits: List[float] = []
    if abs(dx) > 1e-12:
        for edge in (xmin, xmax):
            t = (edge - ax) / dx
            if t > 1e-6:
                y = ay + t * dy
                if ymin - 1e-6 <= y <= ymax + 1e-6:
                    hits.append(t)
    if abs(dy) > 1e-12:
        for edge in (ymin, ymax):
            t = (edge - ay) / dy
            if t > 1e-6:
                x = ax + t * dx
                if xmin - 1e-6 <= x <= xmax + 1e-6:
                    hits.append(t)
    return min(hits) if hits else 0.0


def _cmu_reanchor_radial(
    anno, bbox: Tuple[float, float, float, float], rng: random.Random
) -> Optional[str]:
    if not anno.path:
        return None
    import math

    box = _inner_box(bbox)
    ang = rng.uniform(0.0, 6.283185307)
    dx, dy = math.cos(ang), math.sin(ang)
    tmax = _ray_limit(anno.anchor.x, anno.anchor.y, dx, dy, box)
    if tmax <= 1e-6:
        nx, ny = _clip_xy(anno.path[-1].x, anno.path[-1].y, box)
        start = Point(x=anno.anchor.x, y=anno.anchor.y)
        anno.path = [start, Point(x=nx, y=ny)]
        return f"cmu_radial:{anno.id}:{nx:.1f},{ny:.1f}"
    lo = min(6.0, 0.25 * tmax)
    hi = max(lo, 0.85 * tmax)
    radius = rng.uniform(lo, hi)
    nx = anno.anchor.x + dx * radius
    ny = anno.anchor.y + dy * radius
    nx, ny = _clip_xy(nx, ny, box)
    start = Point(x=anno.anchor.x, y=anno.anchor.y)
    anno.path = [start, Point(x=nx, y=ny)]
    return f"cmu_radial:{anno.id}:{nx:.1f},{ny:.1f}"


def _clamp_state_to_frame(state: LayoutState) -> None:
    box = _inner_box(_view_bbox(state), inset=2.0)
    for lin in state.linears:
        nx, ny = _clip_xy(lin.point.x, lin.point.y, box)
        lin.point = Point(x=nx, y=ny)
        if lin.text_point is not None:
            tx, ty = _clip_xy(lin.text_point.x, lin.text_point.y, box)
            lin.text_point = Point(x=tx, y=ty)
    for anno in list(state.diameters) + list(state.radials):
        for i, pt in enumerate(anno.path or []):
            cx, cy = _clip_xy(pt.x, pt.y, box)
            anno.path[i] = Point(x=cx, y=cy)


_DENSE_MOVE_LO = 2.0
_DENSE_MOVE_HI = 11.0
_DENSE_LIN_FRAC = 0.92
_DENSE_RAD_FRAC = 0.92
_DENSE_MAX_RAD_ANGLE = 0.40  # rad ≈ 23°
_DENSE_LEN_SLACK = 6.0


def text_leader_hits(state: LayoutState) -> int:
    """Count text boxes that intersect another annotation's leader (H_TL)."""
    from layout_agent.geometry_parts import collect_parts, segment_aabb_intersect

    parts = collect_parts(state)
    hits = 0
    ids = list(parts)
    for a in ids:
        crossed = False
        for b in ids:
            if a == b:
                continue
            for seg in parts[b].leaders:
                if segment_aabb_intersect(seg, parts[a].text):
                    hits += 1
                    crossed = True
                    break
            if crossed:
                break
    return hits


def _ann_by_id(state: LayoutState) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for lin in state.linears:
        out[lin.id] = ("linear", lin)
    for anno in list(state.diameters) + list(state.radials):
        out[anno.id] = (anno.category, anno)
    return out


def max_pose_shift_mm(before: LayoutState, after: LayoutState) -> float:
    """Largest Euclidean move of a dim-line point, text, or radial leader end."""
    prev = _ann_by_id(before)
    cur = _ann_by_id(after)
    best = 0.0
    for aid, (kind, now) in cur.items():
        old = prev.get(aid)
        if old is None:
            continue
        _, src = old
        if kind == "linear":
            d = ((now.point.x - src.point.x) ** 2 + (now.point.y - src.point.y) ** 2) ** 0.5
            if d > best:
                best = d
        else:
            a_end = src.path[-1] if src.path else src.anchor
            b_end = now.path[-1] if now.path else now.anchor
            d = ((b_end.x - a_end.x) ** 2 + (b_end.y - a_end.y) ** 2) ** 0.5
            if d > best:
                best = d
    return best


def _snap_linear_text_to_dimline(lin) -> None:
    """Keep text on the dim-line mid-point so outside-text doglegs are not created."""
    from layout_agent.geometry_parts import linear_parts

    parts = linear_parts(lin)
    if parts.dim_lines:
        seg = parts.dim_lines[0]
        lin.text_point = Point(x=0.5 * (seg.a.x + seg.b.x), y=0.5 * (seg.a.y + seg.b.y))
    else:
        lin.text_point = Point(x=lin.point.x, y=lin.point.y)


def _dense_shift_linear(lin, bbox: Tuple[float, float, float, float], rng: random.Random) -> Optional[str]:
    """Slide an existing dim line a short distance along its offset normal."""
    box = _inner_box(bbox)
    nlen = (lin.dir.x * lin.dir.x + lin.dir.y * lin.dir.y) ** 0.5 or 1.0
    ux, uy = lin.dir.x / nlen, lin.dir.y / nlen
    mag = rng.uniform(_DENSE_MOVE_LO, _DENSE_MOVE_HI) * rng.choice([-1.0, 1.0])
    nx, ny = _clip_xy(lin.point.x + ux * mag, lin.point.y + uy * mag, box)
    dx, dy = nx - lin.point.x, ny - lin.point.y
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 0.8:
        return None
    lin.point = Point(x=nx, y=ny)
    _follow_text(lin, dx, dy)
    _snap_linear_text_to_dimline(lin)
    return f"dense_linear:{lin.id}:{dist:.1f}"


def _dense_shift_radial(anno, bbox: Tuple[float, float, float, float], rng: random.Random) -> Optional[str]:
    """Rotate/shorten a radial leader slightly; do not fire it across the sheet."""
    import math

    if not anno.path:
        return None
    box = _inner_box(bbox)
    end = anno.path[-1]
    ax, ay = anno.anchor.x, anno.anchor.y
    old_dx, old_dy = end.x - ax, end.y - ay
    old_len = (old_dx * old_dx + old_dy * old_dy) ** 0.5
    if old_len < 1e-6:
        return None
    ang = math.atan2(old_dy, old_dx) + rng.uniform(-_DENSE_MAX_RAD_ANGLE, _DENSE_MAX_RAD_ANGLE)
    new_len = old_len + rng.uniform(-_DENSE_LEN_SLACK, _DENSE_LEN_SLACK)
    new_len = _clamp(new_len, max(3.0, old_len - _DENSE_MOVE_HI), old_len + _DENSE_MOVE_HI)
    nx = ax + math.cos(ang) * new_len
    ny = ay + math.sin(ang) * new_len
    nx, ny = _clip_xy(nx, ny, box)
    shift = ((nx - end.x) ** 2 + (ny - end.y) ** 2) ** 0.5
    if shift < 0.8:
        return None
    anno.path = [Point(x=ax, y=ay), Point(x=nx, y=ny)]
    return f"dense_radial:{anno.id}:{shift:.1f}"


def _dense_once(state: LayoutState, rng: random.Random) -> Tuple[LayoutState, List[str]]:
    out = state.model_copy(deep=True)
    bbox = _view_bbox(out)
    applied: List[str] = []
    linears = list(out.linears)
    rng.shuffle(linears)
    n_lin = min(len(linears), max(1, int(round(_DENSE_LIN_FRAC * len(linears))))) if linears else 0
    if linears and len(linears) <= 4:
        n_lin = len(linears)
    for lin in linears[:n_lin]:
        note = _dense_shift_linear(lin, bbox, rng)
        if note:
            applied.append(note)
    radials = [a for a in list(out.diameters) + list(out.radials) if a.path]
    rng.shuffle(radials)
    n_rad = min(len(radials), max(1, int(round(_DENSE_RAD_FRAC * len(radials))))) if radials else 0
    if radials and len(radials) <= 3:
        n_rad = len(radials)
    for anno in radials[:n_rad]:
        note = _dense_shift_radial(anno, bbox, rng)
        if note:
            applied.append(note)
    _clamp_state_to_frame(out)
    return out, applied


def perturb_state_dense(
    state: LayoutState,
    rng: random.Random,
) -> Tuple[LayoutState, List[str]]:
    """Move most annotations a short distance; keep text on its dim line / leader end.

    Unlike ``perturb_state_cmu_complex`` this does not side-flip to the far
    frame edge or add extra text jitter (those create long moves and H_TL).
    """
    n_ann = len(state.linears) + len(state.diameters) + len(state.radials)
    orig_tl = text_leader_hits(state)
    min_moved = 1 if n_ann <= 2 else max(2, int(round(0.75 * n_ann)))
    best: Optional[Tuple[Tuple[int, int, float], LayoutState, List[str]]] = None
    for _ in range(20):
        out, applied = _dense_once(state, rng)
        if not applied:
            continue
        if placement_outside_frame(out):
            continue
        if max_pose_shift_mm(state, out) > _DENSE_MOVE_HI + 2.0:
            continue
        tl = text_leader_hits(out)
        if tl > orig_tl:
            continue
        if len(applied) < min_moved:
            continue
        score = (len(applied), -tl, -max_pose_shift_mm(state, out))
        if best is None or score > best[0]:
            best = (score, out, applied)
            if len(applied) >= max(min_moved, int(round(0.9 * n_ann))) and tl <= orig_tl:
                break
    if best is None:
        return state.model_copy(deep=True), []
    return best[1], best[2]


def perturb_state_cmu_complex(
    state: LayoutState,
    rng: random.Random,
) -> Tuple[LayoutState, List[str]]:
    """Complex same-feature perturbation inspired by CMU dim-line randomization.

    Measurement objects (p1/p2/anchor/value) stay fixed. Dim lines, text and
    radial leaders are re-anchored inside the view frame (green bbox), in the
    margin between the measured feature and the frame — side flips and crowding
    stay allowed, long leaders that escape the sheet do not.
    """
    out = state.model_copy(deep=True)
    bbox = _view_bbox(out)
    applied: List[str] = []
    linears = list(out.linears)
    rng.shuffle(linears)
    n_lin = max(3, int(round(0.55 * len(linears)))) if linears else 0
    n_lin = min(len(linears), n_lin)
    for lin in linears[:n_lin]:
        applied.append(_cmu_reanchor_linear(lin, bbox, rng))
    radials = [a for a in list(out.diameters) + list(out.radials) if a.path]
    rng.shuffle(radials)
    n_rad = max(1, int(round(0.6 * len(radials)))) if radials else 0
    n_rad = min(len(radials), n_rad)
    for anno in radials[:n_rad]:
        note = _cmu_reanchor_radial(anno, bbox, rng)
        if note:
            applied.append(note)
    if len(linears) >= 2 and rng.random() < 0.45:
        note = op_swap_layers(out, rng, 6.0, 20.0)
        if note:
            applied.append(note)
    extra_text = min(2, max(0, len(out.linears) - 1))
    for _ in range(extra_text):
        note = op_jitter_linear_text(out, rng, 2.0, 8.0)
        if note:
            applied.append(note)
    _clamp_state_to_frame(out)
    return out, applied


def perturb_state(
    state: LayoutState,
    level: str,
    rng: random.Random,
) -> Tuple[LayoutState, List[str]]:
    """Return (perturbed deep copy, ops applied). Never mutates the input."""
    lo, hi, n_ops = LEVELS[level]
    out = state.model_copy(deep=True)
    applied: List[str] = []
    attempts = 0
    while len(applied) < n_ops and attempts < n_ops * 6:
        attempts += 1
        op = rng.choice(_OPS)
        note = op(out, rng, lo, hi)
        if note:
            applied.append(note)
    return out, applied
