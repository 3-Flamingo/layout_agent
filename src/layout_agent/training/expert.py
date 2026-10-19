"""Deterministic expert and baseline policies for trajectory generation."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from layout_agent.decide import DecisionEngine
from layout_agent.models import Action, ActionType, EvalReport, LayoutState
from layout_agent.training.env import LayoutEnv, enumerate_actions
from layout_agent.training.rl_features import action_key


def _accept_index(actions: Sequence[Action]) -> int:
    return next(
        index
        for index, action in enumerate(actions)
        if action.type == ActionType.ACCEPT
    )


def greedy_action(env: LayoutEnv) -> int:
    """One-step formula oracle; ACCEPT when no legal move improves quality."""
    actions = env.legal_actions
    accept = _accept_index(actions)
    if not env.report.hard_fail:
        return accept
    current = (
        float(env.report.metrics.get("V_hard", 0.0)),
        float(env.report.metrics.get("P", 0.0)),
    )
    candidates: List[Tuple[Tuple[float, float, float, int], int]] = []
    for index, action in enumerate(actions):
        if action.type == ActionType.ACCEPT:
            continue
        _, report, reward, _ = env.simulate_action(index)
        score = (
            float(report.metrics.get("V_hard", 0.0)),
            float(report.metrics.get("P", 0.0)),
            -float(reward),
            index,
        )
        candidates.append((score, index))
    if not candidates:
        return accept
    score, index = min(candidates)
    return index if score[:2] < current else accept


def rule_action(env: LayoutEnv, engine: DecisionEngine) -> int:
    """Map the existing rule DecisionEngine's first legal proposal to a candidate."""
    actions = env.legal_actions
    if not env.report.hard_fail:
        return _accept_index(actions)
    by_key = {action_key(action): index for index, action in enumerate(actions)}
    proposed = engine.decide(env.state, env.report)
    for action in proposed:
        index = by_key.get(action_key(action))
        if index is not None:
            return index
    return greedy_action(env)


@dataclass
class _BeamNode:
    state: LayoutState
    report: EvalReport
    first_key: str
    history: Tuple[str, ...]
    depth: int

    @property
    def score(self) -> Tuple[float, float, int, str]:
        return (
            float(self.report.metrics.get("V_hard", 0.0)),
            float(self.report.metrics.get("P", 0.0)),
            self.depth,
            self.first_key,
        )


def beam_action(env: LayoutEnv, width: int = 4, depth: int = 2) -> int:
    """Finite-width formula beam search over geometry-only transitions."""
    actions = env.legal_actions
    accept = _accept_index(actions)
    if not env.report.hard_fail:
        return accept
    root_score = (
        float(env.report.metrics.get("V_hard", 0.0)),
        float(env.report.metrics.get("P", 0.0)),
    )
    frontier = [
        _BeamNode(
            state=env.executor.snapshot(env.state),
            report=env.report.model_copy(deep=True),
            first_key="",
            history=(),
            depth=0,
        )
    ]
    all_nodes: List[_BeamNode] = []
    for level in range(max(int(depth), 1)):
        expanded: List[_BeamNode] = []
        for node in frontier:
            node_actions = enumerate_actions(
                node.state,
                executor=env.executor,
                history=node.history,
            )
            local: List[_BeamNode] = []
            for action in node_actions:
                if action.type == ActionType.ACCEPT:
                    continue
                candidate, _ = env.executor.apply_geometry(node.state, [action])
                report = env.evaluator.evaluate(candidate)
                key = action_key(action)
                local.append(
                    _BeamNode(
                        state=candidate,
                        report=report,
                        first_key=node.first_key or key,
                        history=(*node.history, key),
                        depth=level + 1,
                    )
                )
            local.sort(key=lambda item: item.score)
            expanded.extend(local[: max(width * 2, width)])
        if not expanded:
            break
        expanded.sort(key=lambda item: item.score)
        frontier = expanded[: max(int(width), 1)]
        all_nodes.extend(frontier)
        if frontier[0].report.hard_fail is False:
            break
    if not all_nodes:
        return greedy_action(env)
    best = min(all_nodes, key=lambda item: item.score)
    if best.score[:2] >= root_score:
        return greedy_action(env)
    for index, action in enumerate(actions):
        if action_key(action) == best.first_key:
            return index
    return greedy_action(env)


class ExpertPolicy:
    """Stateful adapter used by the trajectory collector."""

    def __init__(
        self,
        name: str,
        *,
        cfg: Optional[dict] = None,
        seed: int = 7,
        beam_width: int = 4,
        beam_depth: int = 2,
    ):
        if name not in {"rules", "greedy", "beam", "random"}:
            raise ValueError(f"unknown expert {name!r}")
        self.name = name
        self.rng = random.Random(seed)
        safe_cfg = copy.deepcopy(cfg or {})
        safe_cfg.setdefault("decide", {})["use_llm"] = False
        self.engine = DecisionEngine(safe_cfg)
        self.beam_width = int(beam_width)
        self.beam_depth = int(beam_depth)

    def reset(self, seed: Optional[int] = None) -> None:
        self.engine.reset_search()
        if seed is not None:
            self.rng.seed(int(seed))

    def choose(self, env: LayoutEnv) -> int:
        if self.name == "rules":
            return rule_action(env, self.engine)
        if self.name == "greedy":
            return greedy_action(env)
        if self.name == "beam":
            return beam_action(env, self.beam_width, self.beam_depth)
        legal = list(range(len(env.legal_actions)))
        return legal[self.rng.randrange(len(legal))]
