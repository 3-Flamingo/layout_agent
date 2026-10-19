#!/usr/bin/env python3
"""Validate deterministic replay and optional simulator-vs-CAD coordinates."""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.models import Action, ActionType, LayoutState  # noqa: E402
from layout_agent.training.env import LayoutEnv  # noqa: E402
from layout_agent.training.rl_features import action_key, geometry_fingerprint  # noqa: E402


def _replay_pair(paths: List[Path], episodes: int, seed: int) -> Dict[str, Any]:
    work = Path(tempfile.mkdtemp(prefix="rl_replay_"))
    left = LayoutEnv(paths, work_dir=work / "left", seed=seed)
    right = LayoutEnv(paths, work_dir=work / "right", seed=seed)
    rng = random.Random(seed)
    mismatches: List[Dict[str, Any]] = []
    families: set[str] = set()
    steps = 0
    for episode in range(episodes):
        index = episode % len(paths)
        obs_l, info_l = left.reset(seed=seed + episode, options={"index": index})
        obs_r, info_r = right.reset(seed=seed + episode, options={"index": index})
        if info_l["state_hash"] != info_r["state_hash"]:
            mismatches.append({"episode": episode, "phase": "reset"})
            continue
        frozen = json.dumps(info_l["legal_bbox"], sort_keys=True)
        for _ in range(left.max_steps):
            families.update(action.type.value for action in left.legal_actions)
            candidates = [
                action
                for action in left.legal_actions
                if action.type != ActionType.ACCEPT
            ] or left.legal_actions
            chosen = candidates[rng.randrange(len(candidates))]
            key = action_key(chosen)
            idx_l = next(i for i, action in enumerate(left.legal_actions) if action_key(action) == key)
            idx_r = next(i for i, action in enumerate(right.legal_actions) if action_key(action) == key)
            obs_l, reward_l, term_l, trunc_l, info_l = left.step(idx_l)
            obs_r, reward_r, term_r, trunc_r, info_r = right.step(idx_r)
            steps += 1
            same = (
                reward_l == reward_r
                and (term_l, trunc_l) == (term_r, trunc_r)
                and info_l["state_hash"] == info_r["state_hash"]
                and info_l["termination_reason"] == info_r["termination_reason"]
                and json.dumps(info_l["reward_terms"], sort_keys=True)
                == json.dumps(info_r["reward_terms"], sort_keys=True)
                and np.array_equal(obs_l["action_mask"], obs_r["action_mask"])
                and json.dumps(info_l["legal_bbox"], sort_keys=True) == frozen
                and json.dumps(info_r["legal_bbox"], sort_keys=True) == frozen
            )
            if not same:
                mismatches.append(
                    {
                        "episode": episode,
                        "step": info_l["step_id"],
                        "action": key,
                    }
                )
                break
            if term_l or trunc_l:
                break
    return {
        "episodes": episodes,
        "steps": steps,
        "deterministic": not mismatches,
        "mismatches": mismatches,
        "candidate_action_families": sorted(families),
    }


def _coordinates(state: LayoutState) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for item in state.linears:
        values = [item.point.x, item.point.y, item.dir.x, item.dir.y]
        if item.text_point is not None:
            values.extend([item.text_point.x, item.text_point.y])
        out[item.id] = values
    for item in list(state.diameters) + list(state.radials):
        out[item.id] = [coord for point in item.path for coord in (point.x, point.y)]
    return out


def _coordinate_gap(actual: LayoutState, expected: LayoutState) -> Dict[str, Any]:
    a, b = _coordinates(actual), _coordinates(expected)
    errors: List[float] = []
    missing = sorted(set(a) ^ set(b))
    for ident in sorted(set(a) & set(b)):
        if len(a[ident]) != len(b[ident]):
            missing.append(ident)
            continue
        errors.extend(abs(x - y) for x, y in zip(a[ident], b[ident]))
    return {
        "matched_coordinates": len(errors),
        "missing_or_shape_mismatch": sorted(set(missing)),
        "mean_abs_mm": float(np.mean(errors)) if errors else None,
        "max_abs_mm": float(np.max(errors)) if errors else None,
    }


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _real_gap(path: Path) -> Dict[str, Any]:
    """Compare replay to drawing-2d exports.

    Input rows: {"initial_state": "...", "actions": [{Action...}],
    "expected_state": "drawing-2d exported state.json"}.
    """
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(_load_jsonl(path)):
        initial = Path(record["initial_state"])
        expected = LayoutState.model_validate(
            json.loads(Path(record["expected_state"]).read_text(encoding="utf-8"))
        )
        env = LayoutEnv([initial], use_schedule=False)
        env.reset(options={"index": 0})
        for raw in record.get("actions", []):
            wanted = action_key(Action.model_validate(raw))
            chosen = next(
                (
                    i
                    for i, candidate in enumerate(env.legal_actions)
                    if action_key(candidate) == wanted
                ),
                None,
            )
            if chosen is None:
                raise ValueError(f"row {index}: action is masked: {wanted}")
            _, _, terminated, truncated, _ = env.step(chosen)
            if terminated or truncated:
                break
        rows.append(
            {
                "row": index,
                "state_hash": geometry_fingerprint(env.state),
                **_coordinate_gap(env.state, expected),
            }
        )
    maxima = [row["max_abs_mm"] for row in rows if row["max_abs_mm"] is not None]
    return {
        "samples": rows,
        "worst_max_abs_mm": max(maxima) if maxima else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--real-replays", default="")
    parser.add_argument("--out", default="runs/rl_env_validation.json")
    args = parser.parse_args()
    paths = [Path(value) for value in sorted(glob.glob(args.states, recursive=True))]
    if not paths:
        raise SystemExit(f"no states match {args.states!r}")
    report: Dict[str, Any] = {
        "replay": _replay_pair(paths, args.episodes, args.seed),
    }
    if args.real_replays:
        report["sim_to_real"] = _real_gap(Path(args.real_replays))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["replay"]["deterministic"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
