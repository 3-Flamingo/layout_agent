"""Decompose annotation geometry into constraint parts (text / dim_line / extension / leader)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from layout_agent.models import LayoutState, LinearAnno, Point, RadialAnno


@dataclass
class AABB:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def overlaps(self, other: "AABB") -> bool:
        return not (
            self.xmax < other.xmin
            or other.xmax < self.xmin
            or self.ymax < other.ymin
            or other.ymax < self.ymin
        )

    def inflate(self, pad: float) -> "AABB":
        return AABB(self.xmin - pad, self.ymin - pad, self.xmax + pad, self.ymax + pad)

    def clamp_point_inside(self) -> bool:
        return self.xmin <= self.xmax and self.ymin <= self.ymax

    @property
    def center(self) -> Point:
        return Point(x=0.5 * (self.xmin + self.xmax), y=0.5 * (self.ymin + self.ymax))

    def contains_point(self, p: Point) -> bool:
        return self.xmin <= p.x <= self.xmax and self.ymin <= p.y <= self.ymax


@dataclass
class Segment:
    a: Point
    b: Point

    def length(self) -> float:
        dx = self.b.x - self.a.x
        dy = self.b.y - self.a.y
        return (dx * dx + dy * dy) ** 0.5


@dataclass
class AnnoParts:
    id: str
    category: str  # linear | diameter | radial | chamfer
    text: AABB
    dim_lines: List[Segment] = field(default_factory=list)
    extensions: List[Segment] = field(default_factory=list)
    leaders: List[Segment] = field(default_factory=list)

    def all_points(self) -> List[Point]:
        pts = [
            Point(x=self.text.xmin, y=self.text.ymin),
            Point(x=self.text.xmax, y=self.text.ymax),
        ]
        for seg in self.dim_lines + self.extensions + self.leaders:
            pts.extend([seg.a, seg.b])
        return pts

    def footprint(self) -> AABB:
        pts = self.all_points()
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        return AABB(min(xs), min(ys), max(xs), max(ys))


def _text_aabb(center: Point, tw: float, th: float) -> AABB:
    return AABB(center.x - tw / 2, center.y - th / 2, center.x + tw / 2, center.y + th / 2)


def is_angled_linear(lin: LinearAnno) -> bool:
    """True when dim is not axis-aligned (uses CAD dimAngle or oblique dir)."""
    if lin.dim_angle is not None:
        a = abs(float(lin.dim_angle)) % 90.0
        return min(a, 90.0 - a) > 5.0
    return abs(lin.dir.x) > 0.2 and abs(lin.dir.y) > 0.2


def _dim_angle_rad(lin: LinearAnno) -> float:
    if lin.dim_angle is not None:
        return math.radians(float(lin.dim_angle))
    # fallback: angle of measured edge
    dx, dy = lin.p2.x - lin.p1.x, lin.p2.y - lin.p1.y
    return math.atan2(dy, dx)


def _linear_feet(lin: LinearAnno) -> Tuple[Point, Point, float, float, float]:
    """Return (f1, f2, ux, uy, distance) of the dimension line (CAD projection)."""
    if lin.dim_angle is not None:
        ang = _dim_angle_rad(lin)
        ax, ay = math.cos(ang), math.sin(ang)

        def foot(p: Point) -> Point:
            dx, dy = p.x - lin.point.x, p.y - lin.point.y
            t = dx * ax + dy * ay
            return Point(x=lin.point.x + t * ax, y=lin.point.y + t * ay)

        f1, f2 = foot(lin.p1), foot(lin.p2)
    elif abs(lin.dir.y) >= abs(lin.dir.x):
        f1 = Point(x=lin.p1.x, y=lin.point.y)
        f2 = Point(x=lin.p2.x, y=lin.point.y)
    else:
        f1 = Point(x=lin.point.x, y=lin.p1.y)
        f2 = Point(x=lin.point.x, y=lin.p2.y)
    dist = math.hypot(f2.x - f1.x, f2.y - f1.y)
    if dist < 1e-9:
        return f1, f2, 1.0, 0.0, 0.0
    return f1, f2, (f2.x - f1.x) / dist, (f2.y - f1.y) / dist, dist


def linear_parts(lin: LinearAnno) -> AnnoParts:
    """Decompose linear dim for constraint checks.

    dim_lines = main measurement segment between feet (CAD 尺寸线).
    Arrow stubs / outside-text doglegs / underlines → leaders (参与文字碰撞).
    """
    tw = float(lin.text.get("width", max(6.0, abs(lin.value) * 0.08)))
    th = float(lin.text.get("height", 3.5))
    f1, f2, ux, uy, dist = _linear_feet(lin)
    ext = [Segment(lin.p1, f1), Segment(lin.p2, f2)]
    dim: List[Segment] = [Segment(f1, f2)] if dist > 1e-9 else []
    leaders: List[Segment] = []

    mid = (
        lin.text_point
        if lin.text_point is not None
        else Point(x=0.5 * (f1.x + f2.x), y=0.5 * (f1.y + f2.y))
    )
    aw = float(lin.arrow_width or 2.5)
    arrows_out = dist > 1e-6 and (2.0 * aw + 1.0 > dist)

    # Outside arrow stubs + arrow-body shafts (CAD when arrows cannot fit inside)
    if arrows_out and dist > 1e-9:
        leaders.append(
            Segment(
                Point(x=f1.x - ux * (aw + 3), y=f1.y - uy * (aw + 3)),
                Point(x=f1.x - ux * aw, y=f1.y - uy * aw),
            )
        )
        leaders.append(
            Segment(
                Point(x=f2.x + ux * aw, y=f2.y + uy * aw),
                Point(x=f2.x + ux * (aw + 3), y=f2.y + uy * (aw + 3)),
            )
        )
        # arrow head edges (V) at each foot
        nx, ny = -uy * aw * 0.35, ux * aw * 0.35
        leaders.append(Segment(f1, Point(x=f1.x - ux * aw + nx, y=f1.y - uy * aw + ny)))
        leaders.append(Segment(f1, Point(x=f1.x - ux * aw - nx, y=f1.y - uy * aw - ny)))
        leaders.append(Segment(f2, Point(x=f2.x + ux * aw + nx, y=f2.y + uy * aw + ny)))
        leaders.append(Segment(f2, Point(x=f2.x + ux * aw - nx, y=f2.y + uy * aw - ny)))
    elif dist > 1e-9:
        # Inside arrows: head edges pointing outward from feet
        nx, ny = -uy * aw * 0.35, ux * aw * 0.35
        leaders.append(Segment(f1, Point(x=f1.x + ux * aw + nx, y=f1.y + uy * aw + ny)))
        leaders.append(Segment(f1, Point(x=f1.x + ux * aw - nx, y=f1.y + uy * aw - ny)))
        leaders.append(Segment(f2, Point(x=f2.x - ux * aw + nx, y=f2.y - uy * aw + ny)))
        leaders.append(Segment(f2, Point(x=f2.x - ux * aw - nx, y=f2.y - uy * aw - ny)))

    # Outside text: dogleg + underline as leaders (碰撞检测用)
    if dist > 1e-6 and lin.text_point is not None:
        t_param = (mid.x - f1.x) * ux + (mid.y - f1.y) * uy
        if t_param < -0.5 or t_param > dist + 0.5:
            gap = aw + 3.0
            if t_param < dist * 0.5:
                comb = Point(x=f1.x - ux * gap, y=f1.y - uy * gap)
                leaders.append(Segment(comb, f1))  # dim-line extension toward text
            else:
                comb = Point(x=f2.x + ux * gap, y=f2.y + uy * gap)
                leaders.append(Segment(f2, comb))
            nx, ny = -uy, ux
            feat = Point(x=0.5 * (lin.p1.x + lin.p2.x), y=0.5 * (lin.p1.y + lin.p2.y))
            if (mid.x - feat.x) * nx + (mid.y - feat.y) * ny < 0:
                nx, ny = -nx, -ny
            off = th * 0.55
            half_w = tw / 2.0
            under_a = Point(x=mid.x - ux * half_w + nx * off, y=mid.y - uy * half_w + ny * off)
            under_b = Point(x=mid.x + ux * half_w + nx * off, y=mid.y + uy * half_w + ny * off)
            d0 = math.hypot(under_a.x - comb.x, under_a.y - comb.y)
            d1 = math.hypot(under_b.x - comb.x, under_b.y - comb.y)
            turn = under_a if d0 <= d1 else under_b
            leaders.append(Segment(comb, turn))
            leaders.append(Segment(under_a, under_b))

    return AnnoParts(
        id=lin.id,
        category="linear",
        text=_text_aabb(mid, tw, th),
        dim_lines=dim,
        extensions=ext,
        leaders=leaders,
    )


def radial_parts(dia: RadialAnno) -> AnnoParts:
    tw = float(dia.text.get("width", 10.0))
    th = float(dia.text.get("height", 5.0))
    end = dia.path[-1] if dia.path else dia.anchor
    text = _text_aabb(end, tw, th)
    leaders: List[Segment] = []
    for i in range(len(dia.path) - 1):
        leaders.append(Segment(dia.path[i], dia.path[i + 1]))
    # 尺寸线 = 文字下方短横线
    underline = Segment(
        Point(x=text.xmin, y=text.ymin - 0.4),
        Point(x=text.xmax, y=text.ymin - 0.4),
    )
    return AnnoParts(
        id=dia.id,
        category=dia.category,
        text=text,
        dim_lines=[underline],
        extensions=[],
        leaders=leaders,
    )


def collect_parts(state: LayoutState) -> Dict[str, AnnoParts]:
    out: Dict[str, AnnoParts] = {}
    for lin in state.linears:
        out[lin.id] = linear_parts(lin)
    for dia in list(state.diameters) + list(state.radials):
        out[dia.id] = radial_parts(dia)
    return out


def view_frame(state: LayoutState, pad: float = 5.0) -> AABB:
    # ``legal_bbox`` is immutable during an RL episode. ``render_bbox`` may
    # grow so moved annotations remain visible, but must never relax H_OOB.
    bbox = state.view.get("legal_bbox") or state.view.get("bbox") or {}
    bmin = bbox.get("min", {"x": 0, "y": 0})
    bmax = bbox.get("max", {"x": 100, "y": 80})
    return AABB(
        float(bmin.get("x", 0)) - pad,
        float(bmin.get("y", 0)) - pad,
        float(bmax.get("x", 0)) + pad,
        float(bmax.get("y", 0)) + pad,
    )


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def _on_segment(a: Point, b: Point, c: Point, eps: float = 1e-6) -> bool:
    return (
        min(a.x, b.x) - eps <= c.x <= max(a.x, b.x) + eps
        and min(a.y, b.y) - eps <= c.y <= max(a.y, b.y) + eps
    )


def segments_intersect(s1: Segment, s2: Segment, eps: float = 1e-6) -> bool:
    a, b, c, d = s1.a, s1.b, s2.a, s2.b
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (
        o3 > eps and o4 < -eps or o3 < -eps and o4 > eps
    ):
        return True
    if abs(o1) <= eps and _on_segment(a, b, c, eps):
        return True
    if abs(o2) <= eps and _on_segment(a, b, d, eps):
        return True
    if abs(o3) <= eps and _on_segment(c, d, a, eps):
        return True
    if abs(o4) <= eps and _on_segment(c, d, b, eps):
        return True
    return False


def segment_aabb_intersect(seg: Segment, box: AABB, thickness: float = 0.35) -> bool:
    """Conservative: thicken segment via endpoint boxes + mid sample + AABB edge tests."""
    fat = box.inflate(thickness)
    if fat.contains_point(seg.a) or fat.contains_point(seg.b):
        return True
    # sample midpoints
    for t in (0.25, 0.5, 0.75):
        p = Point(x=seg.a.x + (seg.b.x - seg.a.x) * t, y=seg.a.y + (seg.b.y - seg.a.y) * t)
        if fat.contains_point(p):
            return True
    # segment vs box edges
    corners = [
        Point(x=box.xmin, y=box.ymin),
        Point(x=box.xmax, y=box.ymin),
        Point(x=box.xmax, y=box.ymax),
        Point(x=box.xmin, y=box.ymax),
    ]
    edges = [Segment(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return any(segments_intersect(seg, e) for e in edges)


def point_in_poly(p: Point, poly: Sequence[Point]) -> bool:
    """Ray casting; poly must be closed or will be treated as open ring."""
    if len(poly) < 3:
        return False
    x, y = p.x, p.y
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i].x, poly[i].y
        xj, yj = poly[j].x, poly[j].y
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi):
            inside = not inside
        j = i
    return inside


def iter_obstacle_segments(state: LayoutState) -> Iterable[Tuple[str, Segment]]:
    """Yield (obstacle_id, segment) for every obstacle line (投影线/障碍物线).

    Polyline obstacles yield consecutive segments; loop obstacles also close
    the ring. AABB-only obstacles yield their 4 box edges.
    """
    for ob in state.obstacles:
        pts = ob.polyline or []
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                yield ob.id, Segment(pts[i], pts[i + 1])
            if ob.fill_role == "loop" and len(pts) >= 3:
                yield ob.id, Segment(pts[-1], pts[0])
        elif ob.aabb is not None:
            corners = [
                Point(x=ob.aabb.min.x, y=ob.aabb.min.y),
                Point(x=ob.aabb.max.x, y=ob.aabb.min.y),
                Point(x=ob.aabb.max.x, y=ob.aabb.max.y),
                Point(x=ob.aabb.min.x, y=ob.aabb.max.y),
            ]
            for i in range(4):
                yield ob.id, Segment(corners[i], corners[(i + 1) % 4])
    if state.roughness:
        import math

        from layout_agent.cad_marks import roughness_world_geom

        for mark in state.roughness:
            obstacle_id = f"roughness:{mark.id}"
            geom = roughness_world_geom(mark)
            for polyline in geom.polylines:
                points = [Point(x=x, y=y) for x, y in polyline]
                for index in range(len(points) - 1):
                    yield obstacle_id, Segment(points[index], points[index + 1])
            for cx, cy, radius in geom.circles:
                points = [
                    Point(
                        x=cx + radius * math.cos(2.0 * math.pi * index / 12.0),
                        y=cy + radius * math.sin(2.0 * math.pi * index / 12.0),
                    )
                    for index in range(12)
                ]
                for index in range(12):
                    yield obstacle_id, Segment(points[index], points[(index + 1) % 12])
            for tx, ty, content, _rotation, height in geom.texts:
                half_width = max(len(content), 1) * height * 0.31
                half_height = height * 0.55
                corners = [
                    Point(x=tx - half_width, y=ty - half_height),
                    Point(x=tx + half_width, y=ty - half_height),
                    Point(x=tx + half_width, y=ty + half_height),
                    Point(x=tx - half_width, y=ty + half_height),
                ]
                for index in range(4):
                    yield obstacle_id, Segment(corners[index], corners[(index + 1) % 4])


def text_crosses_obstacle(text: AABB, state: LayoutState) -> Optional[str]:
    """Hard H_TOB: annotation text must not intersect any obstacle line.

    Returns the first offending obstacle id, or None. Complements H_TIN:
    H_TIN catches text fully inside a part loop (no outline crossing);
    this catches text sitting on / crossing projection or obstacle lines.
    """
    for ob_id, seg in iter_obstacle_segments(state):
        if segment_aabb_intersect(seg, text):
            return ob_id
    return None


def text_inside_part(text: AABB, state: LayoutState) -> bool:
    """Soft S3 (near-hard): text center / corners / samples lie in silhouette interior."""
    samples = [
        text.center,
        Point(x=text.xmin, y=text.ymin),
        Point(x=text.xmax, y=text.ymin),
        Point(x=text.xmax, y=text.ymax),
        Point(x=text.xmin, y=text.ymax),
        Point(x=text.center.x, y=text.ymin),
        Point(x=text.center.x, y=text.ymax),
        Point(x=text.xmin, y=text.center.y),
        Point(x=text.xmax, y=text.center.y),
    ]
    for ob in state.obstacles:
        if ob.fill_role != "loop" or not ob.polyline or len(ob.polyline) < 3:
            continue
        hits = sum(1 for p in samples if point_in_poly(p, ob.polyline))
        # majority of samples inside → clearly in-part; also any center hit
        if point_in_poly(text.center, ob.polyline) or hits >= 3:
            return True
    return False
