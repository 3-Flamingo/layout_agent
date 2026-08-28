"""Graded perturbation operators for building weak-supervised preference pairs.

Assumption (validated by the generation script): a random perturbation of a
laid-out view is worse than (or equal to) the original, so
(original, perturbed) forms a preference pair with prefer=original.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

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
