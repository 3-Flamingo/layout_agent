"""Execution: apply discrete actions to geometry, then re-render ICM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from layout_agent.geometry_parts import is_angled_linear
from layout_agent.models import Action, ActionType, LayoutState, LinearAnno, Point
from layout_agent.perceive import Perceiver


class Executor:
    def __init__(self, cfg: Dict[str, Any] | None = None, perceiver: Perceiver | None = None):
        self.cfg = cfg or {}
        self.perceiver = perceiver or Perceiver(cfg)
        ex = (cfg or {}).get("execute", {})
        self.side_gap = float(ex.get("side_gap", 8.0))
        self.layer_step = float(ex.get("layer_step", 4.0))
        # Cap outward layers so dims never shoot past the view frame.
        self.max_layers = int(ex.get("max_layers", 10))
        self.max_gap = float(ex.get("max_gap", 80.0))
        self.hard_nudge_layers = int(ex.get("hard_nudge_layers", 5))
        self.text_shift = float(ex.get("text_shift", 12.0))

    def snapshot(self, state: LayoutState) -> LayoutState:
        return LayoutState.model_validate(state.model_dump(mode="json"))

    def apply(
        self,
        state: LayoutState,
        actions: List[Action],
        work_dir: Path,
    ) -> Tuple[LayoutState, str]:
        """Mutate geometry according to actions and re-render ICM into work_dir."""
        s = self.snapshot(state)
        notes: List[str] = []

        for act in actions:
            if act.type == ActionType.ACCEPT:
                notes.append("accept")
                continue
            if act.type == ActionType.REVERT:
                notes.append("revert_signal")
                continue
            if act.type == ActionType.BALANCE_LINEAR_SIDES:
                s = self._balance_linear_sides(s)
                notes.append("balance_linear_sides")
            elif act.type == ActionType.SET_LINEAR_SIDE and act.side:
                if act.target_id:
                    s = self._set_one_linear_side(s, act.target_id, act.side)
                    notes.append(f"linear_side_{act.target_id}_{act.side}")
                else:
                    # NEVER dump the whole group to one side — balance instead
                    s = self._balance_linear_sides(s, prefer_side=act.side)
                    notes.append(f"linear_side_bulk_redirected_to_balance({act.side})")
            elif act.type == ActionType.SET_RADIAL_SECTOR and act.target_id is not None:
                sector = int(act.sector) if act.sector is not None else None
                s = self._set_radial_sector(s, act.target_id, sector)
                notes.append(f"radial_sector_{act.target_id}_{sector if sector is not None else 'auto'}")
            elif act.type == ActionType.REORDER_RADIAL and act.order:
                notes.append("reorder_noop_geometry")
            elif act.type == ActionType.SET_LINEAR_LAYER_ORDER:
                s = self._push_linears_outward(s, step=self.layer_step)
                notes.append("layer_push")

        s = self._expand_view_to_annos(s)
        inst = work_dir / f"{s.state_id}.instance.png"
        obs = work_dir / f"{s.state_id}.obstacle.png"
        s.state_id = f"{state.state_id}_step"
        s = self.perceiver.render_icm(s, inst, obs)
        return s, ",".join(notes)

    def _expand_view_to_annos(self, state: LayoutState, pad: float = 15.0) -> LayoutState:
        """Grow view bbox so nudged annotations stay inside the frame (clears H_OOB)."""
        from layout_agent.geometry_parts import collect_parts

        parts = collect_parts(state)
        if not parts:
            return state
        xs: List[float] = []
        ys: List[float] = []
        bbox = state.view.get("bbox") or {}
        bmin = bbox.get("min") or {}
        bmax = bbox.get("max") or {}
        xs.extend([float(bmin.get("x", 0)), float(bmax.get("x", 100))])
        ys.extend([float(bmin.get("y", 0)), float(bmax.get("y", 80))])
        for p in parts.values():
            fp = p.footprint()
            xs.extend([fp.xmin, fp.xmax])
            ys.extend([fp.ymin, fp.ymax])
        state.view = {
            **state.view,
            "bbox": {
                "min": {"x": min(xs) - pad, "y": min(ys) - pad},
                "max": {"x": max(xs) + pad, "y": max(ys) + pad},
            },
        }
        return state

    def _view_box(self, state: LayoutState) -> Tuple[float, float, float, float, float, float]:
        bbox = state.view.get("bbox", {})
        bmin, bmax = bbox.get("min", {}), bbox.get("max", {})
        xmin = float(bmin.get("x", 0))
        ymin = float(bmin.get("y", 0))
        xmax = float(bmax.get("x", 0))
        ymax = float(bmax.get("y", 0))
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        return xmin, ymin, xmax, ymax, cx, cy

    def _feature_edges(self, lin: LinearAnno) -> Tuple[float, float, float, float, float, float]:
        """Return (mx, my, x_lo, x_hi, y_lo, y_hi) of the measured edge."""
        mx = 0.5 * (lin.p1.x + lin.p2.x)
        my = 0.5 * (lin.p1.y + lin.p2.y)
        x_lo, x_hi = min(lin.p1.x, lin.p2.x), max(lin.p1.x, lin.p2.x)
        y_lo, y_hi = min(lin.p1.y, lin.p2.y), max(lin.p1.y, lin.p2.y)
        return mx, my, x_lo, x_hi, y_lo, y_hi

    def _text_delta(self, lin: LinearAnno) -> Tuple[float, float]:
        if lin.text_point is None:
            return 0.0, 0.0
        return lin.text_point.x - lin.point.x, lin.text_point.y - lin.point.y

    def _shift_text_further_out(self, lin: LinearAnno) -> None:
        """Push outside text further along the dim-line axis away from the span."""
        if lin.text_point is None:
            return
        from layout_agent.geometry_parts import _linear_feet

        f1, f2, ux, uy, dist = _linear_feet(lin)
        if dist < 1e-9:
            return
        tp = lin.text_point
        t = (tp.x - f1.x) * ux + (tp.y - f1.y) * uy
        amt = self.text_shift
        if t < dist * 0.5:
            # outside toward f1 / negative side
            lin.text_point = Point(x=tp.x - ux * amt, y=tp.y - uy * amt)
        else:
            lin.text_point = Point(x=tp.x + ux * amt, y=tp.y + uy * amt)

    def _set_text_from_delta(self, lin: LinearAnno, dx: float, dy: float) -> None:
        if lin.text_point is None and (dx or dy):
            lin.text_point = Point(x=lin.point.x + dx, y=lin.point.y + dy)
        elif lin.text_point is not None:
            lin.text_point = Point(x=lin.point.x + dx, y=lin.point.y + dy)

    def _place_angled_linear(self, lin: LinearAnno, *, flip: bool, layer: int = 0) -> None:
        """Offset angled dim along its placement normal (dir); optionally flip side."""
        mx, my, _, _, _, _ = self._feature_edges(lin)
        tdx, tdy = self._text_delta(lin)
        dx, dy = lin.dir.x, lin.dir.y
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        ux, uy = dx / norm, dy / norm
        if flip:
            ux, uy = -ux, -uy
        dist0 = abs((lin.point.x - mx) * ux + (lin.point.y - my) * uy)
        want = self._gap_for_layer(layer)
        if flip:
            dist = max(want, self.side_gap + layer * self.layer_step)
        else:
            # Same-side hard nudge: always push further out (layer steps), uncapped by max_gap alone
            bump = max(self.layer_step, max(0, int(layer)) * self.layer_step)
            dist = dist0 + bump
            dist = max(dist, want)
        lin.dir = Point(x=ux, y=uy)
        lin.point = Point(x=mx + ux * dist, y=my + uy * dist)
        self._set_text_from_delta(lin, tdx, tdy)
        if not flip and layer >= self.hard_nudge_layers:
            self._shift_text_further_out(lin)

    def _axis_sides(self, lin: LinearAnno) -> Tuple[str, str]:
        """Legal side pair from *placement axis* (dir), never from p1–p2 span.

        Linear dims only flip within one axis: L↔R or U↔D. Using p1–p2 dx/dy
        wrongly rotates diagonal edges (e.g. vertical length on the right → bottom).
        """
        if abs(lin.dir.x) >= abs(lin.dir.y):
            primary = "L" if lin.dir.x < 0 else "R"
            alternate = "R" if primary == "L" else "L"
            return primary, alternate
        primary = "D" if lin.dir.y < 0 else "U"
        alternate = "U" if primary == "D" else "D"
        return primary, alternate

    def _natural_sides(self, lin: LinearAnno) -> Tuple[str, str]:
        """Alias kept for callers — axis-locked sides."""
        return self._axis_sides(lin)

    def _span_interval(self, lin: LinearAnno, side: str) -> Tuple[float, float]:
        """Projection of the measured edge onto the dim-line axis for this side."""
        _, _, x_lo, x_hi, y_lo, y_hi = self._feature_edges(lin)
        if side in ("U", "D"):
            return (x_lo, x_hi)
        return (y_lo, y_hi)

    @staticmethod
    def _intervals_overlap(a: Tuple[float, float], b: Tuple[float, float], eps: float = 0.5) -> bool:
        return not (a[1] <= b[0] + eps or b[1] <= a[0] + eps)

    def _assign_layers_non_overlapping(
        self, members: List[LinearAnno], side: str
    ) -> Dict[str, int]:
        """Same row (layer) only if projected spans do not intersect.

        Nested / overlapping ranges (e.g. 110 containing 60.03) get different layers.
        Smaller |value| stays inward (lower layer).
        """
        ordered = sorted(members, key=lambda x: abs(x.value))
        layer_spans: List[List[Tuple[float, float]]] = []
        assign: Dict[str, int] = {}
        for lin in ordered:
            iv = self._span_interval(lin, side)
            placed = False
            for li, spans in enumerate(layer_spans):
                if all(not self._intervals_overlap(iv, other) for other in spans):
                    spans.append(iv)
                    assign[lin.id] = li
                    placed = True
                    break
            if not placed:
                assign[lin.id] = len(layer_spans)
                layer_spans.append([iv])
        return assign

    def _gap_for_layer(self, layer: int) -> float:
        layer = max(0, min(int(layer), self.max_layers))
        return min(self.side_gap + layer * self.layer_step, self.max_gap)

    def _place_linear_on_side(
        self,
        lin: LinearAnno,
        side: str,
        *,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        layer: int = 0,
    ) -> None:
        """Place dim line offset from the measured edge; never rotate placement axis."""
        if is_angled_linear(lin):
            # Angled dims: only flip/nudge along existing normal — do not force L/R/U/D axes.
            cur = self._current_side(lin)
            flip = side != cur
            self._place_angled_linear(lin, flip=flip, layer=layer)
            return

        primary, alternate = self._axis_sides(lin)
        if side not in (primary, alternate):
            side = primary

        mx, my, x_lo, x_hi, y_lo, y_hi = self._feature_edges(lin)
        tdx, tdy = self._text_delta(lin)
        # Lateral stagger only when spans are disjoint on the same layer; keep 0 by default
        # to avoid shifting nested dims sideways into each other.
        stagger = 0.0
        pad = 2.0
        want = self._gap_for_layer(layer)

        if side == "D":
            avail = max(self.side_gap * 0.5, y_lo - (ymin + pad))
            y_want = y_lo - min(want, avail)
            # Never collapse inward on hard nudge (keep / increase offset)
            y = min(y_want, lin.point.y) if lin.point.y < y_lo else y_want
            if layer >= self.hard_nudge_layers:
                y = min(y, lin.point.y - self.layer_step)
            lin.point = Point(x=mx + stagger, y=y)
            lin.dir = Point(x=0, y=-1)
        elif side == "U":
            avail = max(self.side_gap * 0.5, (ymax - pad) - y_hi)
            y_want = y_hi + min(want, avail)
            y = max(y_want, lin.point.y) if lin.point.y > y_hi else y_want
            if layer >= self.hard_nudge_layers:
                y = max(y, lin.point.y + self.layer_step)
            lin.point = Point(x=mx + stagger, y=y)
            lin.dir = Point(x=0, y=1)
        elif side == "L":
            avail = max(self.side_gap * 0.5, x_lo - (xmin + pad))
            x_want = x_lo - min(want, avail)
            x = min(x_want, lin.point.x) if lin.point.x < x_lo else x_want
            if layer >= self.hard_nudge_layers:
                x = min(x, lin.point.x - self.layer_step)
            lin.point = Point(x=x, y=my + stagger)
            lin.dir = Point(x=-1, y=0)
        elif side == "R":
            avail = max(self.side_gap * 0.5, (xmax - pad) - x_hi)
            x_want = x_hi + min(want, avail)
            x = max(x_want, lin.point.x) if lin.point.x > x_hi else x_want
            if layer >= self.hard_nudge_layers:
                x = max(x, lin.point.x + self.layer_step)
            lin.point = Point(x=x, y=my + stagger)
            lin.dir = Point(x=1, y=0)
        self._set_text_from_delta(lin, tdx, tdy)

    def _current_side(self, lin: LinearAnno) -> str:
        """Infer side within the locked axis from current point vs feature."""
        if is_angled_linear(lin):
            # Map oblique dir to nearest cardinal for action compatibility.
            if abs(lin.dir.x) >= abs(lin.dir.y):
                return "L" if lin.dir.x < 0 else "R"
            return "D" if lin.dir.y < 0 else "U"
        primary, alternate = self._axis_sides(lin)
        mx, my, _, _, y_lo, y_hi = self._feature_edges(lin)
        if primary in ("U", "D"):
            mid_y = 0.5 * (y_lo + y_hi)
            return "U" if lin.point.y >= mid_y else "D"
        return "R" if lin.point.x >= mx else "L"

    def _balance_linear_sides(
        self,
        state: LayoutState,
        prefer_side: Optional[str] = None,
    ) -> LayoutState:
        """Redistribute within each axis only (L↔R or U↔D); pack non-overlapping rows."""
        xmin, ymin, xmax, ymax, _, _ = self._view_box(state)
        buckets: Dict[str, List[LinearAnno]] = {"L": [], "R": [], "U": [], "D": []}

        for lin in state.linears:
            if is_angled_linear(lin):
                # Keep angled dims out of cardinal L/R/U/D buckets; nudge outward only.
                self._place_angled_linear(lin, flip=False, layer=1)
                continue
            primary, alternate = self._axis_sides(lin)
            side = self._current_side(lin)
            if side not in (primary, alternate):
                side = primary
            # prefer_side only applies if it stays on this dim's axis
            if prefer_side in (primary, alternate):
                pass
            buckets[side].append(lin)

        # Equalize opposite sides *within the same axis only*
        for a, b in (("L", "R"), ("U", "D")):
            na, nb = len(buckets[a]), len(buckets[b])
            if na >= 3 and nb == 0:
                move = buckets[a][na // 2 :]
                buckets[a] = buckets[a][: na // 2]
                buckets[b].extend(move)
            elif nb >= 3 and na == 0:
                move = buckets[b][nb // 2 :]
                buckets[b] = buckets[b][: nb // 2]
                buckets[a].extend(move)
            elif prefer_side == a and nb > na + 1:
                n_move = (nb - na) // 2
                move = buckets[b][-n_move:]
                buckets[b] = buckets[b][:-n_move]
                buckets[a].extend(move)
            elif prefer_side == b and na > nb + 1:
                n_move = (na - nb) // 2
                move = buckets[a][-n_move:]
                buckets[a] = buckets[a][:-n_move]
                buckets[b].extend(move)

        for side, members in buckets.items():
            if not members:
                continue
            # Drop any member whose axis does not match this side (safety)
            legal: List[LinearAnno] = []
            for lin in members:
                primary, alternate = self._axis_sides(lin)
                if side in (primary, alternate):
                    legal.append(lin)
                else:
                    # put back on its primary without rotating
                    self._place_linear_on_side(
                        lin, primary, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, layer=0
                    )
            layer_of = self._assign_layers_non_overlapping(legal, side)
            for lin in legal:
                self._place_linear_on_side(
                    lin,
                    side,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    layer=layer_of.get(lin.id, 0),
                )
        return state

    def _set_one_linear_side(self, state: LayoutState, target_id: str, side: str) -> LayoutState:
        xmin, ymin, xmax, ymax, _, _ = self._view_box(state)
        target = next((x for x in state.linears if x.id == target_id), None)
        if target is None:
            return state

        if is_angled_linear(target):
            cur = self._current_side(target)
            flip = side != cur
            # Same-side request still nudges outward to resolve overlaps.
            layer = 0 if flip else self.hard_nudge_layers
            self._place_angled_linear(target, flip=flip, layer=layer)
            return state

        primary, alternate = self._axis_sides(target)
        if side not in (primary, alternate):
            # Refuse axis rotation (e.g. R → D) — treat as outward nudge on current side
            side = self._current_side(target)
            if side not in (primary, alternate):
                side = primary

        cur = self._current_side(target)
        # Same-side / hard-conflict nudge: push outward without reshuffling others first
        if side == cur:
            self._place_linear_on_side(
                target,
                side,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                layer=self.hard_nudge_layers,
            )
            self._shift_text_further_out(target)
            return state

        on_side_members = []
        for lin in state.linears:
            if lin.id == target_id:
                continue
            if is_angled_linear(lin):
                continue
            p, a = self._axis_sides(lin)
            if side not in (p, a):
                continue
            if self._current_side(lin) == side:
                on_side_members.append(lin)

        # Tentatively place, then re-pack layers on that side including target
        self._place_linear_on_side(
            target, side, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, layer=0
        )
        pack = on_side_members + [target]
        layer_of = self._assign_layers_non_overlapping(pack, side)
        for lin in pack:
            self._place_linear_on_side(
                lin,
                side,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                layer=layer_of.get(lin.id, 0),
            )
        return state

    def _set_linear_side(self, state: LayoutState, side: str) -> LayoutState:
        """Legacy API — redirects to balance to avoid left-pile bug."""
        return self._balance_linear_sides(state, prefer_side=side)

    def _push_linears_outward(self, state: LayoutState, step: float = 3.0) -> LayoutState:
        """Nudge dims outward along their locked axis, then re-pack non-overlapping rows."""
        xmin, ymin, xmax, ymax, _, _ = self._view_box(state)
        sides: Dict[str, List[LinearAnno]] = {"L": [], "R": [], "U": [], "D": []}
        for lin in state.linears:
            side = self._current_side(lin)
            primary, alternate = self._axis_sides(lin)
            if side not in (primary, alternate):
                side = primary
            sides[side].append(lin)
        for side, members in sides.items():
            if not members:
                continue
            layer_of = self._assign_layers_non_overlapping(members, side)
            for lin in members:
                # bump one step outward via higher layer preference for larger values
                base = layer_of.get(lin.id, 0)
                self._place_linear_on_side(
                    lin,
                    side,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    layer=min(base + 1, self.max_layers),
                )
            # re-pack after bump so nested spans don't share a row
            layer_of = self._assign_layers_non_overlapping(members, side)
            for lin in members:
                self._place_linear_on_side(
                    lin,
                    side,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    layer=layer_of.get(lin.id, 0),
                )
        return state

    def _apply_radial_placement(
        self, state: LayoutState, target_id: str, sector: int, length: float
    ) -> None:
        import math

        ang = (sector % 8) * (math.pi / 4)
        dx, dy = math.cos(ang) * length, math.sin(ang) * length
        for lst in (state.diameters, state.radials):
            for d in lst:
                if d.id != target_id:
                    continue
                a = d.anchor
                mid = Point(x=a.x + dx * 0.5, y=a.y + dy * 0.5)
                end = Point(x=a.x + dx, y=a.y + dy)
                d.path = (
                    [a, Point(x=end.x, y=a.y), end]
                    if abs(dx) >= abs(dy)
                    else [a, Point(x=a.x, y=end.y), end]
                )
                if len(d.path) < 2:
                    d.path = [a, mid, end]

    def _set_radial_sector(
        self, state: LayoutState, target_id: str, sector: int | None = None
    ) -> LayoutState:
        """Try multiple sectors/lengths; pick placement minimizing hard then soft cost."""
        from layout_agent.evaluate import Evaluator

        ev = Evaluator(self.cfg)
        preferred = int(sector or 0) % 8
        sector_order = [preferred] + [s for s in range(8) if s != preferred]
        lengths = [12.0, 18.0, 24.0, 30.0]
        best_state = state
        best_key: Optional[Tuple] = None
        for s in sector_order:
            for length in lengths:
                cand = self.snapshot(state)
                self._apply_radial_placement(cand, target_id, s, length)
                rep = ev.evaluate(cand)
                hard_t = sum(
                    1
                    for v in rep.hard_violations
                    if v.a_id == target_id or v.b_id == target_id
                )
                key = (
                    hard_t,
                    float(rep.metrics.get("V_hard", 0)),
                    float(rep.metrics.get("S_text_in_part", 0)),
                    float(rep.metrics.get("S_leader_text", 0)),
                    float(rep.metrics.get("P", 1e18)),
                    length,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_state = cand
                    if (
                        hard_t == 0
                        and float(rep.metrics.get("S_leader_text", 0)) == 0
                        and float(rep.metrics.get("S_text_in_part", 0)) == 0
                    ):
                        return cand
        return best_state
