"""Compare two LayoutStates by movable annotation pose, ignoring ids/colors."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence, Tuple

from layout_agent.models import LayoutState, LinearAnno, RadialAnno


def _r(value: float, nd: int = 2) -> float:
    return round(float(value), nd)


def linear_pose(anno: LinearAnno, nd: int = 2) -> Tuple[Any, ...]:
    text = anno.text_point or anno.point
    return (
        "linear",
        _r(anno.p1.x, nd),
        _r(anno.p1.y, nd),
        _r(anno.p2.x, nd),
        _r(anno.p2.y, nd),
        _r(anno.value, nd),
        _r(anno.point.x, nd),
        _r(anno.point.y, nd),
        _r(text.x, nd),
        _r(text.y, nd),
        _r(anno.dir.x, nd),
        _r(anno.dir.y, nd),
        None if anno.dim_angle is None else _r(float(anno.dim_angle), nd),
    )


def radial_pose(anno: RadialAnno, nd: int = 2) -> Tuple[Any, ...]:
    path = tuple(
        (_r(pt.x - anno.anchor.x, nd), _r(pt.y - anno.anchor.y, nd)) for pt in anno.path
    )
    return (anno.category, _r(anno.anchor.x, nd), _r(anno.anchor.y, nd), path)


def pose_items(state: LayoutState, nd: int = 2) -> Tuple[Tuple[Any, ...], ...]:
    items = [linear_pose(anno, nd) for anno in state.linears]
    items.extend(radial_pose(anno, nd) for anno in state.diameters)
    items.extend(radial_pose(anno, nd) for anno in state.radials)
    return tuple(sorted(items, key=lambda row: json.dumps(row, ensure_ascii=False)))


def layout_pose_fingerprint(state: LayoutState, nd: int = 2) -> str:
    encoded = json.dumps(pose_items(state, nd), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def annotation_count(state: LayoutState) -> int:
    return len(state.linears) + len(state.diameters) + len(state.radials)


def layouts_identical(a: LayoutState, b: LayoutState, nd: int = 2) -> bool:
    return pose_items(a, nd) == pose_items(b, nd)


def pose_delta(a: LayoutState, b: LayoutState, nd: int = 2) -> int:
    """Count of pose tuples that are not in the intersection (symmetric difference)."""
    left = set(pose_items(a, nd))
    right = set(pose_items(b, nd))
    return len(left.symmetric_difference(right))
