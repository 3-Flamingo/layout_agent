"""LayoutEnv: gym-style environment over Executor + Evaluator.

Groundwork for the decision-system RL phase (PPO). The reward follows the
approved plan §2.1B: formula-score delta as the main term, terminal bonus for
clearing hard violations, and a per-step penalty. The learned-quality term is
intentionally NOT wired yet (small-weight hook for later).

Smoke test:
  python -m layout_agent.training.env --states "data/datasets/*/snapshots/*.state.json" --episodes 2
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from layout_agent.config import load_config
from layout_agent.evaluate import Evaluator
from layout_agent.execute import Executor
from layout_agent.models import Action, ActionType, LayoutState
from layout_agent.perceive import Perceiver
from layout_agent.training.data import StateEncoder

_ROOT = Path(__file__).resolve().parents[3]


def enumerate_actions(state: LayoutState) -> List[Action]:
    """Discrete legal action set for the current state (axis-locked sides)."""
    acts: List[Action] = [Action(type=ActionType.ACCEPT, reason="env")]
    acts.append(Action(type=ActionType.BALANCE_LINEAR_SIDES, reason="env"))
    acts.append(Action(type=ActionType.SET_LINEAR_LAYER_ORDER, reason="env"))
    for lin in state.linears:
        sides = ("L", "R") if abs(lin.dir.x) >= abs(lin.dir.y) else ("U", "D")
        for s in sides:
            acts.append(Action(type=ActionType.SET_LINEAR_SIDE, target_id=lin.id, side=s, reason="env"))  # type: ignore[arg-type]
    for a in list(state.diameters) + list(state.radials):
        for sector in range(8):
            acts.append(Action(type=ActionType.SET_RADIAL_SECTOR, target_id=a.id, sector=sector, reason="env"))
    return acts


class LayoutEnv:
    def __init__(
        self,
        state_paths: List[Path],
        max_steps: int = 12,
        img_size: int = 256,
        channels: int = 3,
        work_dir: Optional[Path] = None,
        step_penalty: float = 0.05,
        terminal_bonus: float = 2.0,
        seed: int = 7,
    ):
        cfg = load_config(_ROOT / "config" / "default.yaml")
        self.perceiver = Perceiver(cfg)
        self.evaluator = Evaluator(cfg)
        self.executor = Executor(cfg, self.perceiver)
        self.encoder = StateEncoder(img_size, channels)
        self.state_paths = list(state_paths)
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.terminal_bonus = terminal_bonus
        self.work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="layout_env_"))
        self.rng = random.Random(seed)
        self._state: Optional[LayoutState] = None
        self._p: float = 0.0
        self._steps = 0
        self._actions: List[Action] = []

    # -- helpers -----------------------------------------------------------
    def _obs(self) -> Dict[str, Any]:
        assert self._state is not None
        return {
            "image": self.encoder.encode(self._state).astype(np.float32) / 255.0,
            "state": self._state,
            "legal_actions": self._actions,
        }

    def _eval(self, state: LayoutState) -> Tuple[float, float]:
        rep = self.evaluator.evaluate(state)
        return float(rep.metrics.get("P", 0.0)), float(rep.metrics.get("V_hard", 0.0))

    # -- gym-style API ------------------------------------------------------
    def reset(self, idx: Optional[int] = None) -> Dict[str, Any]:
        path = self.state_paths[idx if idx is not None else self.rng.randrange(len(self.state_paths))]
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        state = LayoutState.model_validate(raw)
        inst = Path(state.raster.instance_image or "")
        if not inst.exists():
            sid = state.state_id or "v"
            state = self.perceiver.render_icm(
                state, self.work / f"{sid}.instance.png", self.work / f"{sid}.obstacle.png"
            )
        self._state = state
        self._p, _ = self._eval(state)
        self._steps = 0
        self._actions = enumerate_actions(state)
        return self._obs()

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert self._state is not None, "call reset() first"
        act = self._actions[action_idx]
        self._steps += 1
        info: Dict[str, Any] = {"action": act.model_dump(mode="json"), "step": self._steps}

        if act.type == ActionType.ACCEPT:
            _, v_hard = self._eval(self._state)
            reward = self.terminal_bonus if v_hard <= 0 else -1.0
            info["v_hard"] = v_hard
            return self._obs(), reward, True, info

        new_state, note = self.executor.apply(self._state, [act], self.work)
        p_new, v_hard = self._eval(new_state)
        reward = (self._p - p_new) / 100.0 - self.step_penalty
        info.update({"note": note, "P": p_new, "v_hard": v_hard})

        self._state = new_state
        self._p = p_new
        self._actions = enumerate_actions(new_state)
        done = self._steps >= self.max_steps
        if done and v_hard <= 0:
            reward += self.terminal_bonus
        return self._obs(), reward, done, info


def _smoke(states_glob: str, episodes: int) -> None:
    import glob

    paths = [Path(p) for p in glob.glob(states_glob)]
    if not paths:
        raise SystemExit(f"no states match: {states_glob}")
    env = LayoutEnv(paths[:20])
    for ep in range(episodes):
        obs = env.reset()
        total = 0.0
        print(f"episode {ep}: state={obs['state'].state_id}, actions={len(obs['legal_actions'])}, "
              f"image={obs['image'].shape}")
        for _ in range(4):
            idx = env.rng.randrange(len(obs["legal_actions"]))
            obs, r, done, info = env.step(idx)
            total += r
            t = info["action"]["type"]
            print(f"  step {info['step']}: {t} r={r:+.3f} P={info.get('P', '-')} done={done}")
            if done:
                break
        print(f"  return: {total:+.3f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    a = ap.parse_args()
    _smoke(a.states, a.episodes)
