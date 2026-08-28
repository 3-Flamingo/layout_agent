"""Constraint-based evaluation (hard / soft) from annotation geometry parts."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from layout_agent.geometry_parts import (
    AnnoParts,
    collect_parts,
    segment_aabb_intersect,
    segments_intersect,
    text_crosses_obstacle,
    text_inside_part,
    view_frame,
)
from layout_agent.models import EvalReport, HardViolation, LayoutState, LinearAnno


class Evaluator:
    """Evaluate layout against 约束文档.md hard/soft constraints.

    Hard → hard_fail / V_hard. Soft → weighted soft cost. P = λ_h·V_hard + λ_s·S.
    Lower P is better.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = (cfg or {}).get("evaluate", cfg or {})
        self.lambda_h = float(self.cfg.get("lambda_h", 100))
        self.lambda_s = float(self.cfg.get("lambda_s", 1))
        # Soft importance (doc scores out of 10) → weights
        soft = self.cfg.get("soft_weights") or {}
        self.w_leader_text = float(soft.get("leader_text", 9.0))
        self.w_layer = float(soft.get("layer_order", 8.0))
        self.w_text_in_part = float(soft.get("text_in_part", 9.5))
        self.w_path = float(soft.get("path_len", 5.0))
        self.w_overcrowd = float(soft.get("overcrowd", 4.0))
        self.w_balance = float(soft.get("side_balance", 3.0))
        self.overcrowd_threshold = float(self.cfg.get("overcrowd_threshold", 3))
        self.frame_pad = float(self.cfg.get("frame_pad", 5.0))
        self.path_scale = float(self.cfg.get("path_scale", 0.02))
        # Near-hard: each in-part text ≈ scale*(w/10)*λ_h (default ~95)
        self.text_in_part_hard_scale = float(self.cfg.get("text_in_part_hard_scale", 0.95))

    def evaluate(self, state: LayoutState) -> EvalReport:
        t0 = time.perf_counter()
        parts = collect_parts(state)
        ids = sorted(parts.keys())
        violations: List[HardViolation] = []
        conflict_pairs: List[Tuple[str, str]] = []
        seen_hard: set = set()

        def add_hard(code: str, a: str, b: str, detail: str = "") -> None:
            key = (code, a, b) if a <= b else (code, b, a)
            # keep directed detail but dedupe undirected pairs per code
            if key in seen_hard:
                return
            seen_hard.add(key)
            violations.append(HardViolation(code=code, a_id=a, b_id=b, detail=detail))
            pair = (a, b)
            if pair not in conflict_pairs and (b, a) not in conflict_pairs:
                conflict_pairs.append(pair)

        # --- Hard 1: text ↔ text ---
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if parts[a].text.overlaps(parts[b].text):
                    add_hard("H_TT", a, b, "text_text")

        # --- Hard 2: text ↔ other dim_line ---
        for i, a in enumerate(ids):
            for b in ids:
                if a == b:
                    continue
                for seg in parts[b].dim_lines:
                    if segment_aabb_intersect(seg, parts[a].text):
                        add_hard("H_TD", a, b, "text_dimline")
                        break

        # --- Hard 3: dim_line ↔ dim_line (different annos) ---
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                hit = False
                for s1 in parts[a].dim_lines:
                    for s2 in parts[b].dim_lines:
                        if segments_intersect(s1, s2) or self._seg_near(s1, s2):
                            add_hard("H_DD", a, b, "dimline_dimline")
                            hit = True
                            break
                    if hit:
                        break

        # --- Hard 4: text ↔ extension (any, including self) ---
        for a in ids:
            for b in ids:
                for seg in parts[b].extensions:
                    # attachment exemption: skip near endpoints of extension (feature side)
                    if self._extension_attachment_exempt(seg, parts[a].text):
                        continue
                    if segment_aabb_intersect(seg, parts[a].text):
                        add_hard("H_TE", a, b, "text_extension")
                        break

        # --- Hard 4b: text ↔ other leaders / arrow stubs ---
        # 箭头外侧短线、文字引出折线、倒角引线 与其它文字硬冲突（原软约束-1 升级）
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                for seg in parts[b].leaders:
                    if segment_aabb_intersect(seg, parts[a].text):
                        add_hard("H_TL", a, b, "text_leader_or_arrow")
                        break

        # --- Hard 5: out of view frame ---
        frame = view_frame(state, pad=self.frame_pad)
        for aid, p in parts.items():
            fp = p.footprint()
            if (
                fp.xmin < frame.xmin
                or fp.xmax > frame.xmax
                or fp.ymin < frame.ymin
                or fp.ymax > frame.ymax
            ):
                add_hard("H_OOB", aid, "frame", "out_of_view_frame")

        # --- Hard 6: text inside part (trial: promoted from soft) ---
        s_text_in = 0.0
        for aid, p in parts.items():
            if text_inside_part(p.text, state):
                s_text_in += 1.0
                add_hard("H_TIN", aid, "part_interior", "text_inside_part")

        # --- Hard 7: text ↔ obstacle line (投影线/障碍物线不得穿字) ---
        s_text_ob = 0.0
        for aid, p in parts.items():
            ob_id = text_crosses_obstacle(p.text, state)
            if ob_id is not None:
                s_text_ob += 1.0
                add_hard("H_TOB", aid, "obstacle", f"text_obstacle_line:{ob_id}")

        # --- Soft metrics ---
        # Leaders already counted as hard H_TL; keep soft counter for UI/compat (= H_TL hits)
        s_leader_text = float(sum(1 for v in violations if v.code == "H_TL"))
        s_layer = float(self._layer_order_penalty(state))

        path_len = 0.0
        for d in list(state.diameters) + list(state.radials):
            for i in range(len(d.path) - 1):
                dx = d.path[i + 1].x - d.path[i].x
                dy = d.path[i + 1].y - d.path[i].y
                path_len += (dx * dx + dy * dy) ** 0.5

        side_density = self._side_density(state)
        overcrowd = sum(max(0.0, v - self.overcrowd_threshold) for v in side_density.values())
        dens_vals = list(side_density.values())
        balance_pen = 0.0
        if dens_vals and max(dens_vals) > 0:
            balance_pen = max(0.0, max(dens_vals) - min(dens_vals) - 1.0)

        soft_cost = (
            # H_TL promoted to hard — do not double-count leader soft
            self.w_layer * s_layer
            + self.w_path * self.path_scale * path_len
            + self.w_overcrowd * overcrowd
            + self.w_balance * balance_pen
        )

        v_hard = float(len(violations))
        p = self.lambda_h * v_hard + self.lambda_s * soft_cost
        hard_fail = v_hard > 0
        dt = (time.perf_counter() - t0) * 1000

        return EvalReport(
            state_id=state.state_id,
            hard_fail=hard_fail,
            hard_violations=violations,
            metrics={
                "P": float(p),
                "V_hard": v_hard,
                "S_total": float(soft_cost),
                "S_leader_text": float(s_leader_text),
                "S_layer": float(s_layer),
                "S_text_in_part": float(s_text_in),  # count; now also in V_hard as H_TIN
                "S_text_in_cost": float(s_text_in * self.lambda_h),  # equivalent hard cost
                "S_text_obstacle": float(s_text_ob),  # count; in V_hard as H_TOB
                "S_path": float(path_len),
                "S_overcrowd": float(overcrowd),
                "S_balance": float(balance_pen),
                # legacy aliases for UI / older runs
                "IF_total": float(s_leader_text + s_text_in),
                "IF_aa": float(s_leader_text),
                "IF_ao": float(s_text_in),
                "Overcrowd": float(overcrowd),
                "PathLen": float(path_len),
            },
            side_density=side_density,
            time_ms=dt,
            conflict_pairs=conflict_pairs,
        )

    @staticmethod
    def _seg_near(s1, s2, tol: float = 0.5) -> bool:
        """Treat nearly coincident overlapping dim lines as interference."""
        # endpoint proximity to other segment AABB-ish
        from layout_agent.geometry_parts import AABB

        def box(s):
            return AABB(
                min(s.a.x, s.b.x) - tol,
                min(s.a.y, s.b.y) - tol,
                max(s.a.x, s.b.x) + tol,
                max(s.a.y, s.b.y) + tol,
            )

        return box(s1).overlaps(box(s2)) and (
            abs((s1.b.x - s1.a.x) * (s2.b.y - s2.a.y) - (s1.b.y - s1.a.y) * (s2.b.x - s2.a.x)) < 1e-3
            or segments_intersect(s1, s2)
        )

    @staticmethod
    def _extension_attachment_exempt(seg, text_box, attach_r: float = 2.0) -> bool:
        """Skip feature-side / dim-line junction noise when text only grazes an endpoint."""
        from layout_agent.geometry_parts import AABB, Point

        mid_p = Point(x=0.5 * (seg.a.x + seg.b.x), y=0.5 * (seg.a.y + seg.b.y))
        if text_box.contains_point(mid_p):
            return False
        cx, cy = text_box.center.x, text_box.center.y
        for end in (seg.a, seg.b):
            zone = AABB(end.x - attach_r, end.y - attach_r, end.x + attach_r, end.y + attach_r)
            if text_box.overlaps(zone) and (cx - end.x) ** 2 + (cy - end.y) ** 2 <= (attach_r * 2.5) ** 2:
                return True
        return False

    @staticmethod
    def _own_underline_exempt(seg, text_box) -> bool:
        """Own underline/dogleg under the label is normal; only interior pierce counts."""
        from layout_agent.geometry_parts import Point

        mid_p = Point(x=0.5 * (seg.a.x + seg.b.x), y=0.5 * (seg.a.y + seg.b.y))
        # Midpoint outside text → underline / approach line, not a glyph pierce
        if not text_box.contains_point(mid_p):
            return True
        return False

    def _side_density(self, state: LayoutState) -> Dict[str, float]:
        side_density = {"L": 0.0, "R": 0.0, "U": 0.0, "D": 0.0}
        for lin in state.linears:
            if abs(lin.dir.x) >= abs(lin.dir.y):
                side_density["L" if lin.dir.x < 0 else "R"] += 1
            else:
                side_density["D" if lin.dir.y < 0 else "U"] += 1
        return side_density

    def _layer_order_penalty(self, state: LayoutState) -> float:
        """Soft-2: same-side parallel linears — non-overlapping may share a row;
        overlapping spans on the same offset are penalized; |value| should grow outward.
        """
        buckets: Dict[str, List[LinearAnno]] = defaultdict(list)
        for lin in state.linears:
            if abs(lin.dir.x) >= abs(lin.dir.y):
                side = "L" if lin.dir.x < 0 else "R"
            else:
                side = "D" if lin.dir.y < 0 else "U"
            buckets[side].append(lin)

        pen = 0.0
        for side, members in buckets.items():
            if len(members) < 2:
                continue
            # group by similar offset (same "row")
            def offset(lin: LinearAnno) -> float:
                if side in ("U", "D"):
                    return round(lin.point.y, 1)
                return round(lin.point.x, 1)

            by_row: Dict[float, List[LinearAnno]] = defaultdict(list)
            for lin in members:
                by_row[offset(lin)].append(lin)

            for row_members in by_row.values():
                for i in range(len(row_members)):
                    for j in range(i + 1, len(row_members)):
                        a, b = row_members[i], row_members[j]
                        ax0, ax1 = min(a.p1.x, a.p2.x), max(a.p1.x, a.p2.x)
                        bx0, bx1 = min(b.p1.x, b.p2.x), max(b.p1.x, b.p2.x)
                        ay0, ay1 = min(a.p1.y, a.p2.y), max(a.p1.y, a.p2.y)
                        by0, by1 = min(b.p1.y, b.p2.y), max(b.p1.y, b.p2.y)
                        if side in ("U", "D"):
                            overlap = not (ax1 <= bx0 + 0.5 or bx1 <= ax0 + 0.5)
                        else:
                            overlap = not (ay1 <= by0 + 0.5 or by1 <= ay0 + 0.5)
                        if overlap:
                            pen += 2.0  # hard-ish soft: nested spans must not share a row

            if side in ("U", "D"):
                ordered = sorted(members, key=lambda x: x.point.y if side == "U" else -x.point.y)
            else:
                ordered = sorted(members, key=lambda x: x.point.x if side == "R" else -x.point.x)
            vals = [abs(x.value) for x in ordered]
            for i in range(len(vals) - 1):
                if vals[i] > vals[i + 1] + 1e-6:
                    pen += 1.0
        return pen
