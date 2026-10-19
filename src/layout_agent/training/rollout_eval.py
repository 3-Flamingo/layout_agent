"""Held-out rollout comparison for learned, rule, greedy and random policies."""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch

from layout_agent.config import load_config
from layout_agent.evalset import perturb_state, split_for, view_key
from layout_agent.evalset.isolation import forbidden_training_views
from layout_agent.models import LayoutState
from layout_agent.training.env import LayoutEnv
from layout_agent.training.expert import ExpertPolicy
from layout_agent.training.policy import (
    CandidatePolicyNet,
    load_policy_checkpoint,
    observation_to_torch,
)

ROOT = Path(__file__).resolve().parents[3]


def _seed(seed: int, key: str, episode: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}:{episode}".encode()).hexdigest()
    return int(digest[:8], 16)


def _rollout(
    env: LayoutEnv,
    initial: LayoutState,
    method: str,
    seed: int,
    *,
    model: CandidatePolicyNet | None,
    device: torch.device,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    observation, info = env.reset(
        seed=seed,
        options={"state": initial, "index": 0},
    )
    expert = None
    if method != "policy":
        expert = ExpertPolicy(method, cfg=cfg, seed=seed)
        expert.reset(seed)
    action_types: Counter[str] = Counter()
    rollbacks = no_ops = 0
    total_reward = 0.0
    for _ in range(env.max_steps + 1):
        if method == "policy":
            assert model is not None
            with torch.no_grad():
                batch = observation_to_torch(observation, device)
                action, _, _, _ = model.act(batch, deterministic=True)
            chosen = int(action.item())
        else:
            assert expert is not None
            chosen = expert.choose(env)
        observation, reward, terminated, truncated, info = env.step(chosen)
        total_reward += reward
        action_types[str(info["action"]["type"])] += 1
        rollbacks += int(bool(info.get("rolled_back")))
        no_ops += int(bool(info.get("no_op")))
        if terminated or truncated:
            break
    return {
        "hard_clear": not env.report.hard_fail,
        "P": float(env.report.metrics.get("P", 0.0)),
        "v_hard": float(env.report.metrics.get("V_hard", 0.0)),
        "steps": int(info["step_id"]),
        "return": float(total_reward),
        "rollbacks": rollbacks,
        "no_ops": no_ops,
        "termination_reason": info.get("termination_reason", ""),
        "action_types": dict(action_types),
    }


def _aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    actions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for row in rows:
        actions.update(row["action_types"])
        reasons[row["termination_reason"]] += 1
    return {
        "episodes": len(rows),
        "hard_clear_rate": float(np.mean([row["hard_clear"] for row in rows]))
        if rows
        else 0.0,
        "mean_P": float(np.mean([row["P"] for row in rows])) if rows else 0.0,
        "mean_v_hard": float(np.mean([row["v_hard"] for row in rows]))
        if rows
        else 0.0,
        "mean_steps": float(np.mean([row["steps"] for row in rows])) if rows else 0.0,
        "mean_return": float(np.mean([row["return"] for row in rows]))
        if rows
        else 0.0,
        "rollback_rate": sum(row["rollbacks"] for row in rows)
        / max(sum(row["steps"] for row in rows), 1),
        "no_op_rate": sum(row["no_ops"] for row in rows)
        / max(sum(row["steps"] for row in rows), 1),
        "action_coverage": dict(actions),
        "termination_reasons": dict(reasons),
    }


def _passes(
    learned: Dict[str, Any],
    baseline: Dict[str, Any],
    *,
    p_tolerance: float,
) -> bool:
    if learned["hard_clear_rate"] > baseline["hard_clear_rate"] + 1e-9:
        return True
    if learned["hard_clear_rate"] + 1e-9 < baseline["hard_clear_rate"]:
        return False
    if learned["mean_P"] < baseline["mean_P"] - p_tolerance:
        return True
    if learned["mean_P"] > baseline["mean_P"] + p_tolerance:
        return False
    return learned["mean_steps"] <= baseline["mean_steps"] + 1e-9


def evaluate_rollouts(
    state_paths: Sequence[Path],
    checkpoint: Path,
    *,
    out: Path,
    split: str = "val",
    val_ratio: float = 0.15,
    levels: Sequence[str] = ("clean", "light", "medium", "heavy"),
    episodes_per_level: int = 1,
    methods: Sequence[str] = ("policy", "rules", "greedy", "random"),
    max_steps: int = 12,
    max_actions: int = 512,
    image_size: int = 256,
    seed: int = 17,
    forbidden: Set[str] | None = None,
    device_name: str = "",
    p_tolerance: float = 0.0,
) -> Dict[str, Any]:
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, checkpoint_meta = load_policy_checkpoint(checkpoint, device=device)
    model.eval()
    cfg = copy.deepcopy(load_config(ROOT / "config" / "default.yaml"))
    cfg.setdefault("decide", {})["use_llm"] = False
    sources: List[Tuple[Path, LayoutState, str]] = []
    seen: set[str] = set()
    for path in sorted(state_paths):
        state = LayoutState.model_validate(json.loads(path.read_text(encoding="utf-8")))
        key = view_key(state)
        if key in seen or key in (forbidden or set()):
            continue
        if split and split_for(key, val_ratio) != split:
            continue
        seen.add(key)
        sources.append((path, state, key))
    if not sources:
        raise ValueError(f"no {split or 'all'} rollout states")

    rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    episode_rows: List[Dict[str, Any]] = []
    work = out.parent / f".{out.stem}_work"
    envs = {
        method: LayoutEnv(
            [sources[0][0]],
            max_steps=max_steps,
            img_size=image_size,
            max_actions=max_actions,
            work_dir=work / method,
            seed=seed,
            forbidden_view_keys=sorted(forbidden or set()),
        )
        for method in methods
    }
    for source_index, (_, state, key) in enumerate(sources):
        for level in levels:
            for repeat in range(episodes_per_level):
                episode = source_index * len(levels) * episodes_per_level + repeat
                episode_seed = _seed(seed, key, episode)
                if level == "clean":
                    initial = state.model_copy(deep=True)
                    perturb_ops: List[str] = []
                else:
                    initial, perturb_ops = perturb_state(
                        state,
                        level,
                        random.Random(episode_seed),
                    )
                for method in methods:
                    result = _rollout(
                        envs[method],
                        initial.model_copy(deep=True),
                        method,
                        episode_seed,
                        model=model,
                        device=device,
                        cfg=cfg,
                    )
                    rows[method].append(result)
                    episode_rows.append(
                        {
                            "view_key": key,
                            "level": level,
                            "repeat": repeat,
                            "method": method,
                            "perturb_ops": perturb_ops,
                            **result,
                        }
                    )
    aggregate = {method: _aggregate(rows[method]) for method in methods}
    baselines = [
        method for method in ("rules", "greedy") if method in aggregate
    ]
    passed = bool(baselines) and all(
        _passes(aggregate["policy"], aggregate[name], p_tolerance=p_tolerance)
        for name in baselines
    )
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.as_posix()),
        "checkpoint_kind": checkpoint_meta.get("kind"),
        "split": split,
        "val_ratio": val_ratio,
        "views": len(sources),
        "levels": list(levels),
        "episodes_per_level": episodes_per_level,
        "aggregate": aggregate,
        "gate": {
            "passed": passed,
            "required_baselines": baselines,
            "rule": "lexicographic hard-clear, mean P, mean steps",
            "p_tolerance": p_tolerance,
        },
        "episodes": episode_rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _forbidden(path: Path, human_gold_dir: Path) -> Set[str]:
    return forbidden_training_views(
        ROOT,
        holdout=path,
        gold_dir=human_gold_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--levels", default="clean,light,medium,heavy")
    parser.add_argument("--episodes-per-level", type=int, default=1)
    parser.add_argument("--methods", default="policy,rules,greedy,random")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="")
    parser.add_argument("--p-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--holdout",
        default=str(ROOT / "data" / "evalsys" / "holdout_views.txt"),
    )
    parser.add_argument(
        "--human-gold-dir",
        default=str(ROOT / "data" / "evalsys" / "human_gold"),
    )
    args = parser.parse_args()
    report = evaluate_rollouts(
        [Path(value) for value in sorted(glob.glob(args.states, recursive=True))],
        Path(args.checkpoint),
        out=Path(args.out),
        split=args.split,
        val_ratio=args.val_ratio,
        levels=[value.strip() for value in args.levels.split(",") if value.strip()],
        episodes_per_level=args.episodes_per_level,
        methods=[value.strip() for value in args.methods.split(",") if value.strip()],
        max_steps=args.max_steps,
        max_actions=args.max_actions,
        image_size=args.image_size,
        seed=args.seed,
        forbidden=_forbidden(
            Path(args.holdout),
            Path(args.human_gold_dir),
        ),
        device_name=args.device,
        p_tolerance=args.p_tolerance,
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))
    if not report["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
