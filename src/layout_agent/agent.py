"""Agent orchestrator: Perceive → Evaluate → Decide → Execute → Brain(schedule) loop."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from layout_agent.decide import DecisionEngine
from layout_agent.evaluate import Evaluator
from layout_agent.execute import Executor
from layout_agent.models import Action, ActionType, AgentResult, LayoutState, ScheduleDecision
from layout_agent.perceive import Perceiver
from layout_agent.schedule import Brain

ProgressCb = Callable[[Dict[str, Any]], None]


class LayoutAgent:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.perceiver = Perceiver(cfg)
        self.evaluator = Evaluator(cfg)
        self.decision = DecisionEngine(cfg)
        self.executor = Executor(cfg, self.perceiver)
        self.brain = Brain(cfg)
        agent_cfg = cfg.get("agent", {})
        self.max_steps = int(agent_cfg.get("max_steps", 5))
        self.timeout_ms = float(agent_cfg.get("timeout_ms", 120000))

    def _progress(
        self,
        cb: Optional[ProgressCb],
        *,
        phase: str,
        step: int,
        total: int,
        message: str,
        **extra: Any,
    ) -> None:
        if not cb:
            return
        pct = int(100 * max(0, min(step, total)) / max(total, 1))
        decide_llm = bool(self.decision.use_llm and self.decision.llm.is_configured())
        brain_llm = bool(self.brain.use_llm and self.brain.llm.is_configured())
        payload: Dict[str, Any] = {
            "phase": phase,
            "step": step,
            "total_steps": total,
            "percent": pct,
            "message": message,
            "use_llm": decide_llm if phase == "decide" else (brain_llm if phase == "schedule" else decide_llm),
            "model": (
                self.brain.llm.model
                if phase == "schedule" and brain_llm
                else (self.decision.llm.model if decide_llm else "rules")
            ),
        }
        payload.update(extra)
        cb(payload)

    def run(
        self,
        state: LayoutState,
        work_dir: Path,
        on_progress: Optional[ProgressCb] = None,
    ) -> AgentResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        self.brain.reset()
        # perceive + eval0 + per-step (decide, execute, evaluate, schedule) + finalize
        total = 2 + self.max_steps * 4 + 1

        self._progress(on_progress, phase="perceive", step=0, total=total, message="感知：加载 / 校验 ICM")
        s0 = self.executor.snapshot(state)
        if not Path(s0.raster.instance_image).exists():
            s0 = self.perceiver.render_icm(
                s0,
                work_dir / f"{s0.state_id}.instance.png",
                work_dir / f"{s0.state_id}.obstacle.png",
            )
        self._progress(on_progress, phase="perceive", step=1, total=total, message="感知完成")

        self._progress(on_progress, phase="evaluate", step=1, total=total, message="评价：计算初始指标 P / IF")
        report0 = self.evaluator.evaluate(s0)
        self._progress(
            on_progress,
            phase="evaluate",
            step=2,
            total=total,
            message=f"初始评价完成 P={report0.metrics.get('P', 0):.2f}",
            metrics=report0.metrics,
            hard_fail=report0.hard_fail,
        )

        best_state = self.executor.snapshot(s0)
        best_report = report0
        best_p = float(report0.metrics.get("P", 1e18))
        applied: List[List[Action]] = []
        schedule_verdicts = []
        reason = "max_steps"
        self.decision.reset_search()
        self.brain.reset()

        cur = s0
        cur_report = report0
        tick = 2

        for step in range(self.max_steps):
            elapsed = (time.perf_counter() - t0) * 1000
            if elapsed > self.timeout_ms:
                reason = "timeout"
                self._progress(
                    on_progress,
                    phase="timeout",
                    step=tick,
                    total=total,
                    message=f"超时停止（{self.timeout_ms:.0f}ms）",
                )
                break

            decide_mode = (
                "LLM"
                if (self.decision.use_llm and self.decision.llm.is_configured())
                else "规则"
            )
            self._progress(
                on_progress,
                phase="decide",
                step=tick,
                total=total,
                message=f"步骤 {step + 1}/{self.max_steps}：决策中（{decide_mode}）…",
                loop_step=step + 1,
            )
            report_before_step = cur_report
            actions = self.decision.decide(cur, cur_report)
            applied.append(actions)
            tick += 1
            action_summary = ", ".join(
                f"{a.type.value}:{a.target_id or a.group_id or a.side or ''}" for a in actions
            ) or "—"
            self._progress(
                on_progress,
                phase="decide",
                step=tick,
                total=total,
                message=f"步骤 {step + 1}：决策结果 → {action_summary}",
                actions=[a.model_dump(mode="json") for a in actions],
                loop_step=step + 1,
            )

            if actions and actions[0].type == ActionType.ACCEPT:
                reason = actions[0].reason or "accept"
                self._progress(
                    on_progress,
                    phase="accept",
                    step=tick,
                    total=total,
                    message=f"决策系统接受当前布局（{reason}）",
                )
                break
            if actions and actions[0].type == ActionType.REVERT:
                reason = "revert"
                break

            self._progress(
                on_progress,
                phase="execute",
                step=tick,
                total=total,
                message=f"步骤 {step + 1}：执行动作并重渲染 ICM…",
                loop_step=step + 1,
            )
            try:
                nxt, note = self.executor.apply(cur, actions, work_dir / f"step_{step}")
            except Exception as e:
                reason = f"exec_error:{e}"
                self._progress(
                    on_progress,
                    phase="error",
                    step=tick,
                    total=total,
                    message=f"执行失败：{e}",
                )
                break
            tick += 1

            self._progress(
                on_progress,
                phase="evaluate",
                step=tick,
                total=total,
                message=f"步骤 {step + 1}：评价新布局…",
                loop_step=step + 1,
                exec_note=note,
            )
            nxt_report = self.evaluator.evaluate(nxt)
            tick += 1
            p1 = float(nxt_report.metrics.get("P", 1e18))
            self._progress(
                on_progress,
                phase="evaluate",
                step=tick,
                total=total,
                message=f"步骤 {step + 1}：P {float(report_before_step.metrics.get('P', 0)):.2f} → {p1:.2f}",
                metrics=nxt_report.metrics,
                loop_step=step + 1,
            )

            # Prefer fewer hard violations, then fewer text-in-part, then lower P
            vh_best = float(best_report.metrics.get("V_hard", 1e9))
            vh_nxt = float(nxt_report.metrics.get("V_hard", 1e9))
            ti_best = float(best_report.metrics.get("S_text_in_part", 1e9))
            ti_nxt = float(nxt_report.metrics.get("S_text_in_part", 1e9))
            better = (vh_nxt, ti_nxt, p1) < (vh_best, ti_best, best_p) or (
                best_report.hard_fail and not nxt_report.hard_fail
            )
            # Never replace a hard-clean best with a hard-failing worse layout
            if nxt_report.hard_fail and not best_report.hard_fail:
                better = False
            if better:
                best_p = p1
                best_state = nxt
                best_report = nxt_report
            else:
                self.decision.note_rejected(actions)
            cur = nxt
            cur_report = nxt_report

            brain_mode = "LLM" if (self.brain.use_llm and self.brain.llm.is_configured()) else "规则"
            self._progress(
                on_progress,
                phase="schedule",
                step=tick,
                total=total,
                message=f"步骤 {step + 1}：调度大脑裁决中（{brain_mode}）…",
                loop_step=step + 1,
            )
            verdict = self.brain.schedule(
                loop_step=step + 1,
                max_steps=self.max_steps,
                report_before_step=report_before_step,
                report_after_step=nxt_report,
                best_p=best_p,
                actions_summary=action_summary,
            )
            schedule_verdicts.append(verdict)
            tick += 1
            self._progress(
                on_progress,
                phase="schedule",
                step=tick,
                total=total,
                message=f"大脑裁决 → {verdict.decision.value}（{verdict.reason}）",
                loop_step=step + 1,
                schedule=verdict.model_dump(mode="json"),
            )

            if verdict.decision == ScheduleDecision.CONTINUE:
                # While hard conflicts remain, keep exploring from the latest state
                # (a short valley may be needed before V_hard drops). Only snap back
                # to best when already hard-clean to protect a good layout.
                if not better and not best_report.hard_fail:
                    cur = best_state
                    cur_report = best_report
                continue
            if verdict.decision == ScheduleDecision.STOP_ACCEPT:
                reason = f"brain:{verdict.reason}"
                break
            if verdict.decision == ScheduleDecision.STOP_NO_IMPROVE:
                reason = f"brain_no_improve:{verdict.reason}"
                break
            if verdict.decision == ScheduleDecision.STOP_REGRESS:
                reason = f"brain_regress:{verdict.reason}"
                break
            if verdict.decision == ScheduleDecision.STOP_MAX:
                reason = "max_steps"
                break

        improved = float(best_report.metrics.get("P", 1e18)) < float(report0.metrics.get("P", 1e18)) or (
            report0.hard_fail and not best_report.hard_fail
        )

        self._progress(
            on_progress,
            phase="finalize",
            step=total - 1,
            total=total,
            message="写出结果文件…",
        )
        self.perceiver.save_state(s0, work_dir / "before.state.json")
        self.perceiver.save_state(best_state, work_dir / "best.state.json")

        result = AgentResult(
            improved=improved,
            steps=len(applied),
            time_ms=(time.perf_counter() - t0) * 1000,
            report_before=report0,
            report_after=best_report,
            actions_applied=applied,
            schedule_verdicts=schedule_verdicts,
            reason=reason,
            state_before_id=s0.state_id,
            state_after_id=best_state.state_id,
        )
        (work_dir / "result.json").write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "report_before.json").write_text(
            json.dumps(report0.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "report_after.json").write_text(
            json.dumps(best_report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        best_img = Path(best_state.raster.instance_image)
        if best_img.exists():
            shutil.copy2(best_img, work_dir / "best.instance.png")
            if best_state.raster.obstacle_image and Path(best_state.raster.obstacle_image).exists():
                shutil.copy2(best_state.raster.obstacle_image, work_dir / "best.obstacle.png")

        self._progress(
            on_progress,
            phase="done",
            step=total,
            total=total,
            message=f"完成：improved={improved} steps={result.steps} reason={reason}",
            result=result.model_dump(mode="json"),
        )
        return result
