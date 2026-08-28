"""Brain / Scheduler: after Execute+Evaluate, decide whether to enter another Decide round."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from layout_agent.llm.client import LlmClient
from layout_agent.models import EvalReport, ScheduleDecision, ScheduleVerdict


class Brain:
    """调度大脑：协调感知/评价/决策/执行闭环，决定是否继续下一轮优化。"""

    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = cfg or {}
        s = self.cfg.get("schedule") or self.cfg.get("brain") or {}
        d = self.cfg.get("decide", {})
        self.use_llm = bool(s.get("use_llm", d.get("use_llm", False)))
        self.temperature = float(s.get("llm_temperature", d.get("llm_temperature", 0.1)))
        self.enable_thinking = bool(s.get("llm_enable_thinking", False))
        self.min_improve = float(s.get("min_improve", 0.5))
        self.patience = int(s.get("patience", 2))  # consecutive no-improve before stop
        self.stop_if_no_hard = bool(s.get("stop_if_no_hard_fail", True))
        self.llm = LlmClient(
            base_url=str(s.get("base_url") or d.get("base_url") or ""),
            api_key=str(s.get("api_key") or d.get("api_key") or ""),
            model=str(s.get("model") or d.get("model") or "deepseek-v4-pro"),
        )
        self._no_improve_streak = 0

    def reset(self) -> None:
        self._no_improve_streak = 0

    def schedule(
        self,
        *,
        loop_step: int,
        max_steps: int,
        report_before_step: EvalReport,
        report_after_step: EvalReport,
        best_p: float,
        actions_summary: str = "",
    ) -> ScheduleVerdict:
        """Return whether to Continue into next Decide, or Stop*."""
        p_before = float(report_before_step.metrics.get("P", 0))
        p_after = float(report_after_step.metrics.get("P", 0))

        # Hard constraints just cleared — accept even on the last step
        if report_before_step.hard_fail and not report_after_step.hard_fail:
            return ScheduleVerdict(
                decision=ScheduleDecision.STOP_ACCEPT,
                reason="hard_fail_cleared",
                loop_step=loop_step,
                p_before=p_before,
                p_after=p_after,
                hard_fail_before=True,
                hard_fail_after=False,
                source="rules",
            )

        if not report_after_step.hard_fail and loop_step >= max_steps:
            return ScheduleVerdict(
                decision=ScheduleDecision.STOP_ACCEPT,
                reason="max_steps_hard_clean",
                loop_step=loop_step,
                p_before=p_before,
                p_after=p_after,
                hard_fail_before=report_before_step.hard_fail,
                hard_fail_after=False,
                source="rules",
            )

        if loop_step >= max_steps:
            return ScheduleVerdict(
                decision=ScheduleDecision.STOP_MAX,
                reason="reached_max_steps",
                loop_step=loop_step,
                p_before=p_before,
                p_after=p_after,
                hard_fail_before=report_before_step.hard_fail,
                hard_fail_after=report_after_step.hard_fail,
                source="rules",
            )

        # While hard_fail remains, use deterministic schedule (LLM often StopRegress too early)
        if report_before_step.hard_fail or report_after_step.hard_fail:
            return self._schedule_rules(
                loop_step=loop_step,
                report_before_step=report_before_step,
                report_after_step=report_after_step,
                best_p=best_p,
            )

        if self.use_llm and self.llm.is_configured():
            try:
                return self._schedule_llm(
                    loop_step=loop_step,
                    max_steps=max_steps,
                    report_before_step=report_before_step,
                    report_after_step=report_after_step,
                    best_p=best_p,
                    actions_summary=actions_summary,
                )
            except Exception as e:
                v = self._schedule_rules(
                    loop_step=loop_step,
                    report_before_step=report_before_step,
                    report_after_step=report_after_step,
                    best_p=best_p,
                )
                v.reason = f"llm_fallback:{e}|{v.reason}"
                return v

        return self._schedule_rules(
            loop_step=loop_step,
            report_before_step=report_before_step,
            report_after_step=report_after_step,
            best_p=best_p,
        )

    def _schedule_rules(
        self,
        *,
        loop_step: int,
        report_before_step: EvalReport,
        report_after_step: EvalReport,
        best_p: float,
    ) -> ScheduleVerdict:
        p0 = float(report_before_step.metrics.get("P", 1e18))
        p1 = float(report_after_step.metrics.get("P", 1e18))
        hf0 = report_before_step.hard_fail
        hf1 = report_after_step.hard_fail
        dens = report_after_step.side_density or {}
        dens_vals = list(dens.values()) if dens else []
        skewed = bool(dens_vals) and max(dens_vals) >= 4 and max(dens_vals) >= 3 * (min(dens_vals) + 0.1)

        # Accept: clean enough and not skewed to one side
        soft_high = (
            float(report_after_step.metrics.get("S_leader_text", 0))
            + float(report_after_step.metrics.get("S_layer", 0))
            + float(report_after_step.metrics.get("S_text_in_part", 0))
        )
        if self.stop_if_no_hard and (not hf1) and soft_high <= 0 and not skewed:
            self._no_improve_streak = 0
            return ScheduleVerdict(
                decision=ScheduleDecision.STOP_ACCEPT,
                reason="no_hard_fail_and_high_soft_clear",
                loop_step=loop_step,
                p_before=p0,
                p_after=p1,
                hard_fail_before=hf0,
                hard_fail_after=hf1,
                source="rules",
            )

        # one-side pile is a layout regress even if P somehow looks better
        dens0 = report_before_step.side_density or {}
        skew0 = bool(dens0) and max(dens0.values()) >= 3 * (min(dens0.values()) + 0.1)
        if skewed and not skew0:
            return ScheduleVerdict(
                decision=ScheduleDecision.STOP_REGRESS,
                reason=f"side_density_skewed:{dens}",
                loop_step=loop_step,
                p_before=p0,
                p_after=p1,
                hard_fail_before=hf0,
                hard_fail_after=hf1,
                source="rules",
            )
        if skewed:
            self._no_improve_streak = 0
            return ScheduleVerdict(
                decision=ScheduleDecision.CONTINUE,
                reason=f"need_rebalance_sides:{dens}",
                loop_step=loop_step,
                p_before=p0,
                p_after=p1,
                hard_fail_before=hf0,
                hard_fail_after=hf1,
                source="rules",
            )

        improved = (p1 + 1e-9) < (best_p - self.min_improve) or (hf0 and not hf1) or (
            int(report_after_step.metrics.get("V_hard", 1e9))
            < int(report_before_step.metrics.get("V_hard", 1e9))
        )
        if improved:
            self._no_improve_streak = 0
            return ScheduleVerdict(
                decision=ScheduleDecision.CONTINUE,
                reason=f"improved_P_or_hard best={best_p:.2f} now={p1:.2f}",
                loop_step=loop_step,
                p_before=p0,
                p_after=p1,
                hard_fail_before=hf0,
                hard_fail_after=hf1,
                source="rules",
            )

        # regression into hard fail from clean
        if (not hf0) and hf1:
            return ScheduleVerdict(
                decision=ScheduleDecision.STOP_REGRESS,
                reason="introduced_hard_fail",
                loop_step=loop_step,
                p_before=p0,
                p_after=p1,
                hard_fail_before=hf0,
                hard_fail_after=hf1,
                source="rules",
            )

        self._no_improve_streak += 1

        # Hard conflicts or near-hard text-in-part remaining: keep trying
        if hf1 or float(report_after_step.metrics.get("S_text_in_part", 0)) > 0:
            return ScheduleVerdict(
                decision=ScheduleDecision.CONTINUE,
                reason=(
                    f"still_hard_or_text_in_part V_hard={report_after_step.metrics.get('V_hard')} "
                    f"S_text_in={report_after_step.metrics.get('S_text_in_part')} "
                    f"streak={self._no_improve_streak}"
                ),
                loop_step=loop_step,
                p_before=p0,
                p_after=p1,
                hard_fail_before=hf0,
                hard_fail_after=hf1,
                source="rules",
            )

        if self._no_improve_streak >= self.patience:
            return ScheduleVerdict(
                decision=ScheduleDecision.STOP_NO_IMPROVE,
                reason=f"no_improve_streak={self._no_improve_streak}",
                loop_step=loop_step,
                p_before=p0,
                p_after=p1,
                hard_fail_before=hf0,
                hard_fail_after=hf1,
                source="rules",
            )

        # still have soft conflicts → keep trying
        if_total = float(report_after_step.metrics.get("S_leader_text", 0)) + float(
            report_after_step.metrics.get("S_layer", 0)
        )
        if if_total > 0:
            return ScheduleVerdict(
                decision=ScheduleDecision.CONTINUE,
                reason=f"still_soft_conflict streak={self._no_improve_streak}",
                loop_step=loop_step,
                p_before=p0,
                p_after=p1,
                hard_fail_before=hf0,
                hard_fail_after=hf1,
                source="rules",
            )

        return ScheduleVerdict(
            decision=ScheduleDecision.STOP_ACCEPT,
            reason="stable_enough",
            loop_step=loop_step,
            p_before=p0,
            p_after=p1,
            hard_fail_before=hf0,
            hard_fail_after=hf1,
            source="rules",
        )

    def _schedule_llm(
        self,
        *,
        loop_step: int,
        max_steps: int,
        report_before_step: EvalReport,
        report_after_step: EvalReport,
        best_p: float,
        actions_summary: str,
    ) -> ScheduleVerdict:
        payload = {
            "loop_step": loop_step,
            "max_steps": max_steps,
            "best_p": best_p,
            "no_improve_streak": self._no_improve_streak,
            "actions": actions_summary,
            "before": {
                "P": report_before_step.metrics.get("P"),
                "V_hard": report_before_step.metrics.get("V_hard"),
                "S_leader_text": report_before_step.metrics.get("S_leader_text"),
                "S_text_in_part": report_before_step.metrics.get("S_text_in_part"),
                "S_total": report_before_step.metrics.get("S_total"),
                "hard_fail": report_before_step.hard_fail,
                "hard_violations": [
                    {"code": v.code, "a": v.a_id, "b": v.b_id}
                    for v in report_before_step.hard_violations[:12]
                ],
            },
            "after": {
                "P": report_after_step.metrics.get("P"),
                "V_hard": report_after_step.metrics.get("V_hard"),
                "S_leader_text": report_after_step.metrics.get("S_leader_text"),
                "S_text_in_part": report_after_step.metrics.get("S_text_in_part"),
                "S_total": report_after_step.metrics.get("S_total"),
                "hard_fail": report_after_step.hard_fail,
                "hard_violations": [
                    {"code": v.code, "a": v.a_id, "b": v.b_id}
                    for v in report_after_step.hard_violations[:12]
                ],
            },
        }
        system = (
            "你是工业图纸标注布局 Agent 的调度大脑（Brain）。"
            "在一轮「决策→执行→评价」之后，决定是否继续进入下一轮决策。"
            "只输出严格 JSON："
            '{"decision":"Continue|StopAccept|StopNoImprove|StopRegress","reason":"..."}。'
            "规则（对齐约束文档）："
            "1) hard_fail=false 且高权重软冲突已很小 → StopAccept；"
            "2) 明显变差（新引入硬冲突、P 大幅上升、或标注被堆到单侧）→ StopRegress；"
            "3) hard_fail=true（含 H_TIN 文字在零件内）时禁止 StopAccept / StopNoImprove，必须 Continue；"
            "4) 仅当 hard_fail=false 且连续无改进 → StopNoImprove；"
            "5) 仍有硬冲突或引线穿字等软冲突且有改进空间 → Continue。"
        )
        text = self.llm.chat_text(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=self.temperature,
            enable_thinking=self.enable_thinking,
        )
        m = re.search(r"\{[\s\S]*\}", text.strip())
        if not m:
            raise ValueError("brain llm: no json")
        data = json.loads(m.group(0))
        raw = str(data.get("decision") or "Continue")
        try:
            decision = ScheduleDecision(raw)
        except Exception:
            # allow lowercase / alias
            alias = {
                "continue": ScheduleDecision.CONTINUE,
                "stopaccept": ScheduleDecision.STOP_ACCEPT,
                "stop": ScheduleDecision.STOP_ACCEPT,
                "accept": ScheduleDecision.STOP_ACCEPT,
                "stopnoimprove": ScheduleDecision.STOP_NO_IMPROVE,
                "stopregress": ScheduleDecision.STOP_REGRESS,
            }
            decision = alias.get(raw.replace("_", "").lower(), ScheduleDecision.CONTINUE)

        p0 = float(report_before_step.metrics.get("P", 0))
        p1 = float(report_after_step.metrics.get("P", 0))
        improved = p1 < best_p - self.min_improve or (
            report_before_step.hard_fail and not report_after_step.hard_fail
        ) or (
            int(report_after_step.metrics.get("V_hard", 1e9))
            < int(report_before_step.metrics.get("V_hard", 1e9))
        )
        # Guardrail: never stop for "no improve" while hard constraints remain
        reason = str(data.get("reason") or "llm")
        if report_after_step.hard_fail and decision in (
            ScheduleDecision.STOP_NO_IMPROVE,
            ScheduleDecision.STOP_ACCEPT,
        ):
            decision = ScheduleDecision.CONTINUE
            reason = f"override_still_hard|{reason}"
        if float(report_after_step.metrics.get("S_text_in_part", 0)) > 0 and decision in (
            ScheduleDecision.STOP_NO_IMPROVE,
            ScheduleDecision.STOP_ACCEPT,
        ):
            decision = ScheduleDecision.CONTINUE
            reason = f"override_text_in_part|{reason}"

        if decision == ScheduleDecision.CONTINUE and improved:
            self._no_improve_streak = 0
        elif decision in (ScheduleDecision.STOP_NO_IMPROVE, ScheduleDecision.CONTINUE) and not improved:
            self._no_improve_streak += 1

        return ScheduleVerdict(
            decision=decision,
            reason=reason,
            loop_step=loop_step,
            p_before=p0,
            p_after=p1,
            hard_fail_before=report_before_step.hard_fail,
            hard_fail_after=report_after_step.hard_fail,
            source="llm",
        )
