"""Deterministic Gymnasium environment for the Decide-only policy.

The policy chooses exactly one legal candidate. Executor, Evaluator and the
rule-only Schedule remain the safety layer. The drawing's legal frame is frozen
at reset; a separate ``render_bbox`` may grow only to keep moved marks visible.
"""

from __future__ import annotations

import copy
import glob
import json
import math
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from layout_agent.config import load_config
from layout_agent.evalset.pairs import view_key as state_view_key
from layout_agent.evaluate import Evaluator
from layout_agent.execute import Executor
from layout_agent.models import (
    Action,
    ActionType,
    EvalReport,
    LayoutState,
    ScheduleDecision,
)
from layout_agent.perceive import Perceiver
from layout_agent.schedule import Brain
from layout_agent.training.data import StateEncoder
from layout_agent.training.rl_features import (
    ACTION_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    action_features,
    action_key,
    dedupe_actions,
    ensure_legal_bbox,
    geometry_fingerprint,
    global_features,
    pad_action_features,
)

_ROOT = Path(__file__).resolve().parents[3]


def _radial_sector(state: LayoutState, target_id: str) -> Optional[int]:
    for anno in list(state.diameters) + list(state.radials):
        if anno.id != target_id or not anno.path:
            continue
        end = anno.path[-1]
        dx, dy = end.x - anno.anchor.x, end.y - anno.anchor.y
        if math.hypot(dx, dy) <= 1e-9:
            return None
        angle = math.atan2(dy, dx) % (2.0 * math.pi)
        return int(round(angle / (math.pi / 4.0))) % 8
    return None


def enumerate_actions(
    state: LayoutState,
    *,
    executor: Optional[Executor] = None,
    history: Optional[Sequence[str]] = None,
) -> List[Action]:
    """Return stable, de-duplicated candidates with invalid/no-op moves removed."""
    attempted = set(history or ())
    actions: List[Action] = [Action(type=ActionType.ACCEPT, reason="env")]
    if state.linears:
        actions.extend(
            [
                Action(type=ActionType.BALANCE_LINEAR_SIDES, reason="env"),
                Action(type=ActionType.SET_LINEAR_LAYER_ORDER, reason="env"),
            ]
        )
    for linear in sorted(state.linears, key=lambda item: item.id):
        if abs(linear.dir.x) >= abs(linear.dir.y):
            current = "L" if linear.dir.x < 0 else "R"
            alternatives = ("L", "R")
        else:
            current = "D" if linear.dir.y < 0 else "U"
            alternatives = ("U", "D")
        for side in alternatives:
            if side == current:
                continue
            actions.append(
                Action(
                    type=ActionType.SET_LINEAR_SIDE,
                    target_id=linear.id,
                    side=side,  # type: ignore[arg-type]
                    reason="env",
                )
            )
    for radial in sorted(
        list(state.diameters) + list(state.radials),
        key=lambda item: (item.category, item.id),
    ):
        current = _radial_sector(state, radial.id)
        for sector in range(8):
            if sector == current:
                continue
            actions.append(
                Action(
                    type=ActionType.SET_RADIAL_SECTOR,
                    target_id=radial.id,
                    sector=sector,
                    reason="env",
                )
            )

    actions = [
        action
        for action in dedupe_actions(actions)
        if action.type == ActionType.ACCEPT or action_key(action) not in attempted
    ]
    if executor is None:
        return actions

    # Global operations can already be satisfied. Detect their no-op geometry
    # before exposing them to the policy.
    before = geometry_fingerprint(state)
    filtered: List[Action] = []
    for action in actions:
        if action.type in {
            ActionType.BALANCE_LINEAR_SIDES,
            ActionType.SET_LINEAR_LAYER_ORDER,
        }:
            candidate, _ = executor.apply_geometry(state, [action])
            if geometry_fingerprint(candidate) == before:
                continue
        filtered.append(action)
    return filtered


class LayoutEnv(gym.Env[Dict[str, np.ndarray], int]):
    """Candidate-action Gymnasium environment with deterministic replay."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        state_paths: Sequence[Path | str],
        max_steps: int = 12,
        img_size: int = 256,
        channels: int = 3,
        max_actions: int = 512,
        work_dir: Optional[Path] = None,
        step_penalty: float = 0.01,
        no_op_penalty: float = 0.25,
        hard_clear_bonus: float = 1.0,
        accept_clean_bonus: float = 2.0,
        accept_hard_penalty: float = 1.0,
        rollback_penalty: float = 0.25,
        seed: int = 7,
        use_schedule: bool = True,
        forbidden_view_keys: Optional[Sequence[str]] = None,
        config_path: Optional[Path] = None,
        human_reward_checkpoint: Optional[Path] = None,
        human_reward_gate: Optional[Path] = None,
        human_reward_beta: float = 0.0,
        human_reward_device: Optional[str] = None,
    ):
        super().__init__()
        if not state_paths:
            raise ValueError("state_paths cannot be empty")
        cfg = copy.deepcopy(load_config(config_path or (_ROOT / "config" / "default.yaml")))
        cfg.setdefault("decide", {})["use_llm"] = False
        cfg.setdefault("schedule", {})["use_llm"] = False
        self.perceiver = Perceiver(cfg)
        self.evaluator = Evaluator(cfg)
        self.executor = Executor(cfg, self.perceiver)
        self.brain = Brain(cfg)
        self.encoder = StateEncoder(img_size, channels)
        self.state_paths = [Path(path) for path in state_paths]
        self.max_steps = int(max_steps)
        self.img_size = int(img_size)
        self.channels = int(channels)
        self.max_actions = int(max_actions)
        self.step_penalty = float(step_penalty)
        self.no_op_penalty = float(no_op_penalty)
        self.hard_clear_bonus = float(hard_clear_bonus)
        self.accept_clean_bonus = float(accept_clean_bonus)
        self.accept_hard_penalty = float(accept_hard_penalty)
        self.rollback_penalty = float(rollback_penalty)
        self.use_schedule = bool(use_schedule)
        self.forbidden_view_keys = set(forbidden_view_keys or ())
        self._human_reward = None
        if human_reward_beta > 0:
            if human_reward_checkpoint is None or human_reward_gate is None:
                raise ValueError(
                    "positive human_reward_beta requires checkpoint and gate"
                )
            from layout_agent.training.human_reward import GatedHumanReward

            self._human_reward = GatedHumanReward.load(
                Path(human_reward_checkpoint),
                Path(human_reward_gate),
                beta=human_reward_beta,
                device=human_reward_device,
            )
        self.work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="layout_env_"))
        self.work.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self._initial_seed = int(seed)
        self._episode_counter = 0
        self._episode_id = ""
        self._state: Optional[LayoutState] = None
        self._report: Optional[EvalReport] = None
        self._best_state: Optional[LayoutState] = None
        self._best_report: Optional[EvalReport] = None
        self._steps = 0
        self._actions: List[Action] = []
        self._action_history: List[str] = []
        self._terminated = False
        self._truncated = False

        self.action_space = spaces.Discrete(self.max_actions)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.channels, self.img_size, self.img_size),
                    dtype=np.float32,
                ),
                "global_features": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(GLOBAL_FEATURE_DIM,),
                    dtype=np.float32,
                ),
                "action_features": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.max_actions, ACTION_FEATURE_DIM),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(self.max_actions),
            }
        )

    @property
    def state(self) -> LayoutState:
        if self._state is None:
            raise RuntimeError("call reset() first")
        return self._state

    @property
    def report(self) -> EvalReport:
        if self._report is None:
            raise RuntimeError("call reset() first")
        return self._report

    @property
    def legal_actions(self) -> List[Action]:
        return list(self._actions)

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def _episode_dir(self, step: Optional[int] = None) -> Path:
        suffix = "reset" if step is None else f"step_{step:03d}"
        path = self.work / self._episode_id / suffix
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _render(self, state: LayoutState, step: Optional[int] = None) -> LayoutState:
        sid = state.state_id or "state"
        path = self._episode_dir(step)
        return self.perceiver.render_icm(
            state,
            path / f"{sid}.instance.png",
            path / f"{sid}.obstacle.png",
        )

    def _refresh_actions(self) -> None:
        self._actions = enumerate_actions(
            self.state,
            executor=self.executor,
            history=self._action_history,
        )
        if len(self._actions) > self.max_actions:
            raise ValueError(
                f"{len(self._actions)} legal actions exceed max_actions={self.max_actions}"
            )

    def _observation(self) -> Dict[str, np.ndarray]:
        feats = action_features(self.state, self.report, self._actions)
        padded, mask = pad_action_features(feats, self.max_actions)
        observation = {
            "image": self.encoder.encode(self.state).astype(np.float32) / 255.0,
            "global_features": global_features(
                self.state,
                self.report,
                self._steps,
                self.max_steps,
            ),
            "action_features": padded,
            "action_mask": mask,
        }
        return observation

    def _info(self, **extra: Any) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "episode_id": self._episode_id,
            "step_id": self._steps,
            "state_id": self.state.state_id,
            "state_hash": geometry_fingerprint(self.state),
            "view_key": state_view_key(self.state),
            "legal_action_count": len(self._actions),
            "legal_actions": [action.model_dump(mode="json") for action in self._actions],
            "P": float(self.report.metrics.get("P", 0.0)),
            "v_hard": float(self.report.metrics.get("V_hard", 0.0)),
            "hard_fail": self.report.hard_fail,
            "legal_bbox": copy.deepcopy(
                self.state.view.get("legal_bbox") or self.state.view.get("bbox")
            ),
            "render_bbox": copy.deepcopy(self.state.view.get("render_bbox")),
        }
        info.update(extra)
        return info

    @staticmethod
    def _quality(report: EvalReport) -> Tuple[float, float]:
        return (
            float(report.metrics.get("V_hard", 0.0)),
            float(report.metrics.get("P", 0.0)),
        )

    def _formula_reward(
        self,
        before: EvalReport,
        after: EvalReport,
        *,
        no_op: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        hard_delta = float(before.metrics.get("V_hard", 0.0)) - float(
            after.metrics.get("V_hard", 0.0)
        )
        soft_delta = (
            float(before.metrics.get("S_total", 0.0))
            - float(after.metrics.get("S_total", 0.0))
        ) / 100.0
        clear_bonus = (
            self.hard_clear_bonus
            if before.hard_fail and not after.hard_fail
            else 0.0
        )
        no_op_cost = self.no_op_penalty if no_op else 0.0
        terms = {
            "hard_delta": hard_delta,
            "soft_delta": soft_delta,
            "hard_clear_bonus": clear_bonus,
            "step_penalty": -self.step_penalty,
            "no_op_penalty": -no_op_cost,
        }
        return sum(terms.values()), terms

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(int(seed))
        options = options or {}
        if "state" in options:
            raw_state = options["state"]
            state = (
                LayoutState.model_validate(raw_state)
                if isinstance(raw_state, dict)
                else LayoutState.model_validate(raw_state.model_dump(mode="json"))
            )
            source_idx = int(options.get("index", -1))
        else:
            source_idx = int(
                options.get("index", self.rng.randrange(len(self.state_paths)))
            )
            path = self.state_paths[source_idx]
            state = LayoutState.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        view_key = state_view_key(state)
        if view_key in self.forbidden_view_keys:
            raise ValueError(f"training reset attempted frozen view_key: {view_key}")

        ensure_legal_bbox(state)
        state.view.pop("render_bbox", None)
        self._episode_counter += 1
        seed_tag = self._initial_seed if seed is None else int(seed)
        self._episode_id = (
            f"ep{self._episode_counter:06d}_seed{seed_tag}_src{source_idx}"
        )
        self._steps = 0
        self._action_history = []
        self._terminated = False
        self._truncated = False
        self.brain.reset()
        self._state = self._render(state)
        self._report = self.evaluator.evaluate(self._state)
        self._best_state = self.executor.snapshot(self._state)
        self._best_report = self._report.model_copy(deep=True)
        self._refresh_actions()
        return self._observation(), self._info(reset=True, source_index=source_idx)

    def simulate_action(
        self,
        action_idx: int,
    ) -> Tuple[LayoutState, EvalReport, float, Dict[str, Any]]:
        """Evaluate one candidate without mutating the episode (expert look-ahead)."""
        if action_idx < 0 or action_idx >= len(self._actions):
            raise ValueError(f"masked/invalid action index: {action_idx}")
        action = self._actions[action_idx]
        if action.type == ActionType.ACCEPT:
            reward = (
                self.accept_clean_bonus
                if not self.report.hard_fail
                else -self.accept_hard_penalty
            )
            return self.executor.snapshot(self.state), self.report.model_copy(deep=True), reward, {
                "accept": reward
            }
        candidate, _ = self.executor.apply_geometry(self.state, [action])
        ensure_legal_bbox(candidate)
        report = self.evaluator.evaluate(candidate)
        no_op = geometry_fingerprint(candidate) == geometry_fingerprint(self.state)
        reward, terms = self._formula_reward(self.report, report, no_op=no_op)
        return candidate, report, reward, terms

    def step(
        self,
        action_idx: int,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("call reset() first")
        if self._terminated or self._truncated:
            raise RuntimeError("episode is done; call reset()")
        if action_idx < 0 or action_idx >= len(self._actions):
            raise ValueError(f"masked/invalid action index: {action_idx}")

        started = time.perf_counter()
        action = self._actions[action_idx]
        self._steps += 1
        self._action_history.append(action_key(action))
        action_dump = action.model_dump(mode="json")
        if action.type == ActionType.ACCEPT:
            reward = (
                self.accept_clean_bonus
                if not self.report.hard_fail
                else -self.accept_hard_penalty
            )
            self._terminated = True
            return (
                self._observation(),
                reward,
                True,
                False,
                self._info(
                    action=action_dump,
                    termination_reason=(
                        "accept_clean" if not self.report.hard_fail else "accept_with_hard"
                    ),
                    reward_terms={"accept": reward},
                    step_time_ms=(time.perf_counter() - started) * 1000.0,
                ),
            )

        before_state = self.state
        before_report = self.report
        legal_before = copy.deepcopy(
            before_state.view.get("legal_bbox") or before_state.view.get("bbox")
        )
        new_state, note = self.executor.apply(
            before_state,
            [action],
            self._episode_dir(self._steps),
        )
        ensure_legal_bbox(new_state)
        legal_after = new_state.view.get("legal_bbox") or new_state.view.get("bbox")
        if legal_after != legal_before:
            raise RuntimeError("Executor modified immutable legal_bbox")
        new_report = self.evaluator.evaluate(new_state)
        no_op = geometry_fingerprint(new_state) == geometry_fingerprint(before_state)
        reward, reward_terms = self._formula_reward(
            before_report,
            new_report,
            no_op=no_op,
        )
        if self._human_reward is not None:
            human_delta, human_terms = self._human_reward.delta(
                before_state,
                new_state,
                before_report,
                new_report,
            )
            reward += human_delta
            reward_terms["human_reward"] = human_delta
            reward_terms.update(human_terms)

        prior_best_p = float(
            (self._best_report or before_report).metrics.get("P", 0.0)
        )
        if self._best_report is None or self._quality(new_report) < self._quality(
            self._best_report
        ):
            self._best_state = self.executor.snapshot(new_state)
            self._best_report = new_report.model_copy(deep=True)

        schedule = None
        termination_reason = ""
        terminated = False
        truncated = False
        if self.use_schedule:
            schedule = self.brain.schedule(
                loop_step=self._steps,
                max_steps=self.max_steps,
                report_before_step=before_report,
                report_after_step=new_report,
                best_p=prior_best_p,
                actions_summary=note,
            )
            if schedule.decision == ScheduleDecision.STOP_ACCEPT:
                terminated = True
                termination_reason = schedule.reason
            elif schedule.decision in {
                ScheduleDecision.STOP_REGRESS,
                ScheduleDecision.STOP_NO_IMPROVE,
            }:
                terminated = True
                termination_reason = schedule.reason
                reward -= self.rollback_penalty
                reward_terms["rollback_penalty"] = -self.rollback_penalty
                if self._best_state is not None and self._best_report is not None:
                    new_state = self.executor.snapshot(self._best_state)
                    new_report = self._best_report.model_copy(deep=True)
            elif schedule.decision == ScheduleDecision.STOP_MAX:
                truncated = True
                termination_reason = schedule.reason
                if self._best_state is not None and self._best_report is not None:
                    new_state = self.executor.snapshot(self._best_state)
                    new_report = self._best_report.model_copy(deep=True)

        if not terminated and not truncated and self._steps >= self.max_steps:
            truncated = True
            termination_reason = "max_steps"
            if self._best_state is not None and self._best_report is not None:
                new_state = self.executor.snapshot(self._best_state)
                new_report = self._best_report.model_copy(deep=True)

        self._state = new_state
        self._report = new_report
        self._terminated = terminated
        self._truncated = truncated
        self._refresh_actions()
        info = self._info(
            action=action_dump,
            note=note,
            no_op=no_op,
            reward_terms=reward_terms,
            termination_reason=termination_reason,
            schedule=(schedule.model_dump(mode="json") if schedule else None),
            rolled_back=bool(
                schedule
                and schedule.decision
                in {
                    ScheduleDecision.STOP_REGRESS,
                    ScheduleDecision.STOP_NO_IMPROVE,
                    ScheduleDecision.STOP_MAX,
                }
            ),
            step_time_ms=(time.perf_counter() - started) * 1000.0,
        )
        return self._observation(), float(reward), terminated, truncated, info


def _smoke(states_glob: str, episodes: int) -> None:
    paths = [Path(path) for path in glob.glob(states_glob)]
    if not paths:
        raise SystemExit(f"no states match: {states_glob}")
    env = LayoutEnv(paths[:20])
    for episode in range(episodes):
        obs, info = env.reset(seed=7 + episode)
        total = 0.0
        print(
            f"episode {episode}: state={info['state_id']}, "
            f"actions={info['legal_action_count']}, image={obs['image'].shape}"
        )
        for _ in range(4):
            legal = np.flatnonzero(obs["action_mask"])
            idx = int(env.rng.choice(list(legal)))
            obs, reward, terminated, truncated, info = env.step(idx)
            total += reward
            print(
                f"  step {info['step_id']}: {info['action']['type']} "
                f"r={reward:+.3f} P={info['P']:.2f} "
                f"done={terminated or truncated}"
            )
            if terminated or truncated:
                break
        print(f"  return: {total:+.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()
    _smoke(args.states, args.episodes)
