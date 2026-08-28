"""Decision: rule-based + optional LLM (deepseek-v4-pro)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set

from layout_agent.llm.client import LlmClient
from layout_agent.models import Action, ActionType, EvalReport, LayoutState


# Prefer resolving harder / more specific violations first
_HARD_PRIORITY = {
    "H_TT": 0,
    "H_TIN": 1,
    "H_TOB": 2,
    "H_TD": 3,
    "H_TL": 4,
    "H_TE": 5,
    "H_DD": 6,
    "H_OOB": 7,
}


class DecisionEngine:
    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = cfg or {}
        d = self.cfg.get("decide", {})
        self.use_llm = bool(d.get("use_llm", False))
        # One-shot 全局规划：为所有硬冲突生成动作批（配合执行侧逐动作复评筛选）。
        # 默认关闭以保持迭代式 agent 循环（每轮小步 + taboo 轮换）的原有行为。
        self.global_plan = bool(d.get("global_plan", False))
        self.max_plan_actions = int(d.get("max_plan_actions", 16))
        self.temperature = float(d.get("llm_temperature", 0.2))
        self.enable_thinking = bool(d.get("llm_enable_thinking", False))
        self.llm = LlmClient(
            base_url=str(d.get("base_url", "")),
            api_key=str(d.get("api_key", "")),
            model=str(d.get("model", "deepseek-v4-pro")),
        )
        self._sector_cursor = 0
        self._strategy = 0
        self._taboo: Set[tuple] = set()

    def reset_search(self) -> None:
        self._strategy = 0
        self._taboo.clear()
        self._sector_cursor = 0

    def note_rejected(self, actions: List[Action]) -> None:
        """Record a failed action batch so the next decide tries a different strategy."""
        self._taboo.add(self._action_sig(actions))
        self._strategy += 1

    @staticmethod
    def _action_sig(actions: List[Action]) -> tuple:
        return tuple(
            (a.type.value if hasattr(a.type, "value") else str(a.type), a.target_id, a.side, a.sector)
            for a in actions
        )

    def _side_for(self, lin, *, prefer_flip: bool) -> str:
        if abs(lin.dir.x) >= abs(lin.dir.y):
            primary, alternate = ("L", "R") if lin.dir.x < 0 else ("R", "L")
        else:
            primary, alternate = ("D", "U") if lin.dir.y < 0 else ("U", "D")
        return alternate if prefer_flip else primary

    def decide(self, state: LayoutState, report: EvalReport) -> List[Action]:
        soft_ok = (
            float(report.metrics.get("S_leader_text", 0)) <= 0
            and float(report.metrics.get("S_text_in_part", 0)) <= 0
            and float(report.metrics.get("S_overcrowd", report.metrics.get("Overcrowd", 0))) <= 0
            and float(report.metrics.get("S_layer", 0)) <= 0
        )
        if (not report.hard_fail) and soft_ok:
            return [Action(type=ActionType.ACCEPT, reason="no_hard_or_high_soft")]

        # Hard conflicts: prefer deterministic outward-nudge rules (LLM often flips
        # illegal axes and regresses P → 优化对比显示移动 0).
        if report.hard_fail:
            actions = self._decide_rules_entry(state, report)
            if actions and actions[0].type != ActionType.ACCEPT:
                return actions

        actions: List[Action] = []
        if self.use_llm and self.llm.is_configured():
            try:
                actions = self._decide_llm(state, report)
                actions = self._sanitize_actions(actions)
                for a in actions:
                    if not a.reason or a.reason == "llm":
                        a.reason = a.reason or "llm"
                    elif not a.reason.startswith("llm"):
                        a.reason = f"llm:{a.reason}"
            except Exception as e:
                actions = self._decide_rules_entry(state, report)
                if actions:
                    actions[0].reason = f"llm_fallback:{e}"
        else:
            actions = self._decide_rules_entry(state, report)

        if not actions:
            return [Action(type=ActionType.ACCEPT, reason="no_action")]
        return actions

    def _sanitize_actions(self, actions: List[Action]) -> List[Action]:
        """Prevent bulk SetLinearSide without target_id (causes left-pile)."""
        out: List[Action] = []
        for a in actions:
            # empty side from LLM breaks pydantic — drop it
            side = a.side if a.side in ("L", "R", "U", "D") else None
            if a.type == ActionType.SET_LINEAR_SIDE and not a.target_id:
                out.append(
                    Action(
                        type=ActionType.BALANCE_LINEAR_SIDES,
                        reason=(a.reason or "llm") + "|sanitized_no_bulk_side",
                    )
                )
            else:
                out.append(
                    Action(
                        type=a.type,
                        group_id=a.group_id,
                        side=side,
                        target_id=a.target_id,
                        sector=a.sector,
                        order=a.order,
                        reason=a.reason,
                    )
                )
        return out

    def _ids_from_violations(self, report: EvalReport) -> List[str]:
        ordered = sorted(
            report.hard_violations,
            key=lambda v: (_HARD_PRIORITY.get(v.code, 9), v.code, v.a_id, v.b_id),
        )
        ids: List[str] = []
        seen: Set[str] = set()
        for v in ordered:
            for tid in (v.a_id, v.b_id):
                if tid in ("frame", "obstacle", "part_interior", "-"):
                    continue
                if tid not in seen:
                    seen.add(tid)
                    ids.append(tid)
        if not ids:
            for a, b in report.conflict_pairs:
                for tid in (a, b):
                    if tid in ("frame", "obstacle", "part_interior", "-"):
                        continue
                    if tid not in seen:
                        seen.add(tid)
                        ids.append(tid)
        return ids

    def _decide_rules_entry(self, state: LayoutState, report: EvalReport) -> List[Action]:
        if self.global_plan:
            actions = self._decide_rules_global(state, report)
            if actions:
                return actions
        return self._decide_rules(state, report)

    def _decide_rules_global(self, state: LayoutState, report: EvalReport) -> List[Action]:
        """One-shot 全局规划：覆盖全部硬冲突，而非首个规则命中即返回。

        与迭代式 _decide_rules 的区别：
        - 先给全局动作（BalanceLinearSides / SetLinearLayerOrder）整体重排；
        - 再为每条硬冲突涉及的标注生成针对性动作（每个标注最多一次）；
        - 动作可以宽松多给：执行侧应逐动作复评，丢弃使布局变差的动作。
        """
        radial_ids = {d.id for d in state.diameters} | {r.id for r in state.radials}
        linear_ids = {x.id for x in state.linears}
        dens = report.side_density or {}
        n_hard = len(report.hard_violations)
        actions: List[Action] = []

        # 全局动作先行：多条冲突时一次性重排/分层通常能消掉大部分同侧堆叠
        skewed = bool(dens) and max(dens.values()) >= 2 * max(1.0, min(dens.values()) + 0.1)
        if n_hard >= 2 or float(report.metrics.get("S_layer", 0)) > 0 or skewed:
            actions.append(
                Action(type=ActionType.BALANCE_LINEAR_SIDES, reason="global_rebalance")
            )
            actions.append(
                Action(type=ActionType.SET_LINEAR_LAYER_ORDER, reason="global_layer_push")
            )

        ordered = sorted(
            report.hard_violations,
            key=lambda v: (_HARD_PRIORITY.get(v.code, 9), v.code, v.a_id, v.b_id),
        )
        prefer_flip = self._strategy % 2 == 1
        seen: Set[tuple] = set()
        for v in ordered:
            for tid in (v.a_id, v.b_id):
                if tid in ("frame", "obstacle", "part_interior", "-", None):
                    continue
                if tid in radial_ids:
                    if ("sector", tid) in seen:
                        continue
                    seen.add(("sector", tid))
                    self._sector_cursor = (self._sector_cursor + 1) % 8
                    actions.append(
                        Action(
                            type=ActionType.SET_RADIAL_SECTOR,
                            target_id=tid,
                            sector=self._sector_cursor,
                            reason=f"global_{v.code.lower()}_radial",
                        )
                    )
                elif tid in linear_ids:
                    if ("side", tid) in seen:
                        continue
                    seen.add(("side", tid))
                    lin = next(x for x in state.linears if x.id == tid)
                    side = self._side_for(lin, prefer_flip=prefer_flip)
                    actions.append(
                        Action(
                            type=ActionType.SET_LINEAR_SIDE,
                            target_id=tid,
                            side=side,  # type: ignore
                            reason=f"global_{v.code.lower()}_move",
                        )
                    )
        if not actions and float(
            report.metrics.get("S_overcrowd", report.metrics.get("Overcrowd", 0))
        ) > 0:
            actions.append(
                Action(type=ActionType.BALANCE_LINEAR_SIDES, reason="global_overcrowd")
            )
        return actions[: self.max_plan_actions]

    def _decide_rules(self, state: LayoutState, report: EvalReport) -> List[Action]:
        actions: List[Action] = []
        radial_ids = {d.id for d in state.diameters} | {r.id for r in state.radials}
        linear_ids = {x.id for x in state.linears}
        targets = self._ids_from_violations(report)

        for tid in targets:
            if tid in radial_ids:
                self._sector_cursor = (self._sector_cursor + 1) % 8
                actions.append(
                    Action(
                        type=ActionType.SET_RADIAL_SECTOR,
                        target_id=tid,
                        sector=self._sector_cursor,
                        reason="rule_resolve_radial_hard",
                    )
                )
                return actions

        # Text-text hard: nudge or flip involved linears (strategy rotates on reject)
        prefer_flip = self._strategy % 2 == 1
        tt_done: Set[str] = set()
        for v in report.hard_violations:
            if v.code != "H_TT":
                continue
            for tid in (v.a_id, v.b_id):
                if tid not in linear_ids or tid in tt_done:
                    continue
                lin = next(x for x in state.linears if x.id == tid)
                side = self._side_for(lin, prefer_flip=prefer_flip)
                actions.append(
                    Action(
                        type=ActionType.SET_LINEAR_SIDE,
                        target_id=tid,
                        side=side,  # type: ignore
                        reason="rule_tt_flip" if prefer_flip else "rule_tt_nudge_out",
                    )
                )
                tt_done.add(tid)
                if prefer_flip:
                    # flip one at a time
                    break
            if actions and self._action_sig(actions) not in self._taboo:
                return actions
            actions = []

        # Text-extension / text-dimline / text-leader / text-obstacle: nudge or flip pierced text
        te_done: Set[str] = set()
        for v in report.hard_violations:
            if v.code not in ("H_TE", "H_TD", "H_TL", "H_TOB"):
                continue
            tid = v.a_id if v.a_id in linear_ids else None
            if tid is None or tid in te_done:
                continue
            lin = next(x for x in state.linears if x.id == tid)
            side = self._side_for(lin, prefer_flip=prefer_flip)
            actions.append(
                Action(
                    type=ActionType.SET_LINEAR_SIDE,
                    target_id=tid,
                    side=side,  # type: ignore
                    reason=f"rule_{v.code.lower()}_{'flip' if prefer_flip else 'nudge_out'}",
                )
            )
            te_done.add(tid)
        if actions and self._action_sig(actions) not in self._taboo:
            return actions
        if actions and self._action_sig(actions) in self._taboo:
            # last resort: flip opposite strategy for TE batch
            actions = []
            te_done = set()
            for v in report.hard_violations:
                if v.code not in ("H_TE", "H_TD", "H_TL", "H_TOB"):
                    continue
                tid = v.a_id if v.a_id in linear_ids else None
                if tid is None or tid in te_done:
                    continue
                lin = next(x for x in state.linears if x.id == tid)
                side = self._side_for(lin, prefer_flip=not prefer_flip)
                actions.append(
                    Action(
                        type=ActionType.SET_LINEAR_SIDE,
                        target_id=tid,
                        side=side,  # type: ignore
                        reason=f"rule_{v.code.lower()}_retry",
                    )
                )
                te_done.add(tid)
            if actions and self._action_sig(actions) not in self._taboo:
                return actions

        if float(report.metrics.get("S_text_in_part", 0)) > 0:
            in_part: List[str] = []
            for a, b in report.conflict_pairs:
                if b == "part_interior" and a not in in_part:
                    in_part.append(a)
                elif a == "part_interior" and b not in in_part:
                    in_part.append(b)
            for tid in in_part:
                if tid in radial_ids:
                    self._sector_cursor = (self._sector_cursor + 1) % 8
                    actions.append(
                        Action(
                            type=ActionType.SET_RADIAL_SECTOR,
                            target_id=tid,
                            sector=self._sector_cursor,
                            reason="rule_text_out_of_part_radial",
                        )
                    )
                    return actions
            # Linears inside part: rebalance to push outward on legal sides
            if any(tid in linear_ids for tid in in_part):
                actions.append(
                    Action(
                        type=ActionType.BALANCE_LINEAR_SIDES,
                        reason="rule_text_out_of_part_balance",
                    )
                )
                return actions
            for tid in in_part:
                if tid in linear_ids:
                    lin = next(x for x in state.linears if x.id == tid)
                    if abs(lin.dir.x) >= abs(lin.dir.y):
                        cur = "L" if lin.dir.x < 0 else "R"
                        side = "R" if cur == "L" else "L"
                    else:
                        cur = "D" if lin.dir.y < 0 else "U"
                        side = "U" if cur == "D" else "D"
                    actions.append(
                        Action(
                            type=ActionType.SET_LINEAR_SIDE,
                            target_id=tid,
                            side=side,  # type: ignore
                            reason="rule_text_out_of_part_flip",
                        )
                    )
                    return actions

        dens = report.side_density or {}
        if float(report.metrics.get("S_layer", 0)) > 0 or (
            dens and max(dens.values()) >= 3 * max(1.0, min(dens.values()) + 0.1)
        ):
            actions.append(
                Action(
                    type=ActionType.BALANCE_LINEAR_SIDES,
                    reason="rule_balance_layer_or_skew",
                )
            )
            return actions

        for tid in targets:
            if tid in linear_ids:
                lin = next(x for x in state.linears if x.id == tid)
                # Flip only within locked axis (dir), never rotate R→D
                if abs(lin.dir.x) >= abs(lin.dir.y):
                    cur = "L" if lin.dir.x < 0 else "R"
                    side = "R" if cur == "L" else "L"
                else:
                    cur = "D" if lin.dir.y < 0 else "U"
                    side = "U" if cur == "D" else "D"
                actions.append(
                    Action(
                        type=ActionType.SET_LINEAR_SIDE,
                        target_id=tid,
                        side=side,  # type: ignore
                        reason="rule_move_conflict_linear",
                    )
                )
                return actions

        if float(report.metrics.get("S_overcrowd", report.metrics.get("Overcrowd", 0))) > 0:
            actions.append(
                Action(type=ActionType.BALANCE_LINEAR_SIDES, reason="rule_overcrowd")
            )
        return actions

    def _decide_llm(self, state: LayoutState, report: EvalReport) -> List[Action]:
        dens = report.side_density or {}
        summary = {
            "hard_fail": report.hard_fail,
            "hard_violations": [
                {"code": v.code, "a": v.a_id, "b": v.b_id, "detail": v.detail}
                for v in report.hard_violations[:24]
            ],
            "metrics": report.metrics,
            "side_density": dens,
            "conflicts": report.conflict_pairs[:20],
            "constraint_notes": {
                "hard": [
                    "H_TT text-text",
                    "H_TIN text-inside-part",
                    "H_TOB text-obstacle_line",
                    "H_TD text-other_dimline",
                    "H_TL text-leader_or_arrow",
                    "H_DD dimline-dimline",
                    "H_TE text-extension",
                    "H_OOB out_of_frame",
                ],
                "soft": [
                    "S_leader_text(9)",
                    "S_layer(8)",
                    "path(5)",
                    "overcrowd(4)",
                    "balance(3)",
                ],
                "priority": "先消硬约束（含文字不得在零件内 H_TIN）；再降软代价",
            },
            "linears": [
                {
                    "id": x.id,
                    "value": x.value,
                    "point": x.point.model_dump(),
                    "p1": x.p1.model_dump(),
                    "p2": x.p2.model_dump(),
                    "dir": x.dir.model_dump(),
                    "dim_angle": x.dim_angle,
                    "text_point": x.text_point.model_dump() if x.text_point else None,
                    "angled": bool(
                        x.dim_angle is not None
                        and min(abs(float(x.dim_angle)) % 90.0, 90.0 - (abs(float(x.dim_angle)) % 90.0)) > 5.0
                    ),
                }
                for x in state.linears
            ],
            "diameters": [
                {"id": x.id, "anchor": x.anchor.model_dump(), "path_end": (x.path[-1].model_dump() if x.path else None)}
                for x in state.diameters
            ],
            "radials": [
                {
                    "id": x.id,
                    "category": x.category,
                    "anchor": x.anchor.model_dump(),
                    "path_end": (x.path[-1].model_dump() if x.path else None),
                    "content": (x.text or {}).get("content"),
                }
                for x in state.radials
            ],
        }
        system = (
            "你是工业图纸单视图标注布局决策器，必须遵守约束文档。"
            "硬约束必须消除：文字-文字、文字不得在零件内部(H_TIN)、文字不得压投影线/障碍物线(H_TOB)、文字-他标尺寸线、尺寸线-尺寸线、文字-尺寸界线、不出视图外框。"
            "软约束按权重：引线-他文字(9)、同侧分层并排(8)、引线短(5)、不过密(4)、四向均匀(3)。"
            "只输出严格 JSON："
            '{"actions":[{"type":"SetRadialSector|SetLinearSide|BalanceLinearSides|SetLinearLayerOrder|Accept",'
            '"target_id":"...","sector":0,"side":"L|R|U|D","reason":"..."}]}。'
            "约束："
            "1) SetLinearSide 必须带 target_id 与 side（L|R|U|D），只移动单个冲突标注；不要输出空 side；"
            "   轴对齐尺寸：只能在原轴内翻转（L↔R 或 U↔D）；"
            "   angled=true 的斜向尺寸：side 表示沿其法向翻到对侧/外推，禁止拧成另一正交轴；"
            "2) 文字重叠(H_TT)先 SetLinearLayerOrder 再对其中一个 SetLinearSide；"
            "3) 单侧过密用 BalanceLinearSides；同侧投影区间有交集的尺寸禁止同一排；"
            "4) 直径/半径/倒角(chamfer)硬冲突优先 SetRadialSector（sector 0-7）；"
            "5) H_TIN / H_TOB（文字压投影线）必须优先外推；"
            "6) 仅当 hard_fail=false 时 Accept。"
        )
        user = "当前评价与状态摘要：\n" + json.dumps(summary, ensure_ascii=False)
        text = self.llm.chat_text(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            enable_thinking=self.enable_thinking,
        )
        return self._parse_actions_json(text)

    def _parse_actions_json(self, text: str) -> List[Action]:
        text = text.strip()
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError("no json object in llm output")
        data = json.loads(m.group(0))
        raw = data.get("actions") or []
        out: List[Action] = []
        for item in raw:
            t = item.get("type")
            try:
                at = ActionType(t)
            except Exception:
                continue
            side = item.get("side")
            if side not in ("L", "R", "U", "D"):
                side = None
            out.append(
                Action(
                    type=at,
                    group_id=item.get("group_id"),
                    side=side,
                    target_id=item.get("target_id"),
                    sector=item.get("sector"),
                    order=item.get("order"),
                    reason=item.get("reason") or "llm",
                )
            )
        return out
