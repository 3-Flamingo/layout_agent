"""Collect rule/greedy/beam/random trajectories for the Decide-only policy."""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from layout_agent.config import load_config
from layout_agent.evalset import perturb_state, split_for, view_key
from layout_agent.evalset.isolation import forbidden_training_views
from layout_agent.models import LayoutState
from layout_agent.training.env import LayoutEnv
from layout_agent.training.expert import ExpertPolicy
from layout_agent.training.trajectory import (
    TrajectoryShardWriter,
    summarize_trajectories,
)

ROOT = Path(__file__).resolve().parents[3]
EXPERTS = ("rules", "greedy", "beam", "random")
LEVELS = ("clean", "light", "medium", "heavy")


def _read_state(path: Path) -> LayoutState:
    return LayoutState.model_validate(json.loads(path.read_text(encoding="utf-8")))


def forbidden_views(
    holdout: Path,
    human_gold_dir: Path,
    *,
    root: Optional[Path] = None,
) -> Set[str]:
    return forbidden_training_views(
        root or ROOT,
        holdout=holdout,
        gold_dir=human_gold_dir,
    )


def _episode_seed(seed: int, view: str, episode: int) -> int:
    digest = hashlib.sha256(f"{seed}:{view}:{episode}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _prepare_state(
    state: LayoutState,
    level: str,
    seed: int,
) -> Tuple[LayoutState, List[str]]:
    if level == "clean":
        return state.model_copy(deep=True), []
    return perturb_state(state, level, random.Random(seed))


def collect(
    state_paths: Sequence[Path],
    out: Path,
    *,
    episodes_per_state: int,
    experts: Sequence[str],
    levels: Sequence[str],
    seed: int,
    val_ratio: float,
    max_steps: int,
    max_actions: int,
    image_size: int,
    shard_size: int,
    forbidden: Set[str],
    beam_width: int,
    beam_depth: int,
) -> Dict[str, object]:
    cfg = copy.deepcopy(load_config(ROOT / "config" / "default.yaml"))
    cfg.setdefault("decide", {})["use_llm"] = False
    unique: List[Tuple[Path, LayoutState, str]] = []
    seen_views: set[str] = set()
    for path in sorted(state_paths):
        try:
            state = _read_state(path)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {path}: {exc}")
            continue
        key = view_key(state)
        if key in forbidden or key in seen_views:
            continue
        if not (state.linears or state.diameters or state.radials):
            continue
        seen_views.add(key)
        unique.append((path, state, key))
    if not unique:
        raise ValueError("no leak-free, actionable source states")

    env = LayoutEnv(
        [path for path, _, _ in unique],
        max_steps=max_steps,
        img_size=image_size,
        max_actions=max_actions,
        work_dir=out / "_env_work",
        seed=seed,
        forbidden_view_keys=sorted(forbidden),
    )
    policies = {
        name: ExpertPolicy(
            name,
            cfg=cfg,
            seed=seed,
            beam_width=beam_width,
            beam_depth=beam_depth,
        )
        for name in experts
    }
    combinations = [(name, level) for name in experts for level in levels]
    if not combinations:
        raise ValueError("experts and levels cannot be empty")

    with TrajectoryShardWriter(out, shard_size=shard_size) as writer:
        for source_index, (path, source, key) in enumerate(unique):
            split = split_for(key, val_ratio)
            for episode in range(episodes_per_state):
                expert_name, level = combinations[episode % len(combinations)]
                episode_seed = _episode_seed(seed, key, episode)
                initial, perturb_ops = _prepare_state(source, level, episode_seed)
                policy = policies[expert_name]
                policy.reset(episode_seed)
                observation, info = env.reset(
                    seed=episode_seed,
                    options={"state": initial, "index": source_index},
                )
                for _ in range(max_steps + 1):
                    state_hash = info["state_hash"]
                    chosen = policy.choose(env)
                    chosen_action = env.legal_actions[chosen].model_dump(mode="json")
                    next_observation, reward, terminated, truncated, next_info = env.step(chosen)
                    done = bool(terminated or truncated)
                    writer.add(
                        observation,
                        chosen_action=chosen,
                        reward=reward,
                        done=done,
                        split=split,
                        expert=expert_name,
                        level=level,
                        episode_id=env.episode_id,
                        view_key=key,
                        metadata={
                            "source_state": str(path.as_posix()),
                            "perturb_ops": perturb_ops,
                            "state_hash": state_hash,
                            "next_state_hash": next_info["state_hash"],
                            "action": chosen_action,
                            "reward_terms": next_info.get("reward_terms", {}),
                            "termination_reason": next_info.get("termination_reason", ""),
                        },
                    )
                    observation, info = next_observation, next_info
                    if done:
                        break
    summary = summarize_trajectories(out)
    summary.update(
        {
            "source_views": len(unique),
            "forbidden_views": len(forbidden),
            "episodes_per_state": episodes_per_state,
            "experts": list(experts),
            "levels": list(levels),
            "seed": seed,
            "val_ratio": val_ratio,
        }
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes-per-state", type=int, default=16)
    parser.add_argument("--experts", default="rules,greedy,beam,random")
    parser.add_argument("--levels", default="clean,light,medium,heavy")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--beam-depth", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--holdout",
        default=str(ROOT / "data" / "evalsys" / "holdout_views.txt"),
    )
    parser.add_argument(
        "--human-gold-dir",
        default=str(ROOT / "data" / "evalsys" / "human_gold"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.glob("shard_*.npz")):
        if not args.overwrite:
            raise SystemExit(f"{out} already has shards; pass --overwrite")
        shutil.rmtree(out)
    paths = [Path(value) for value in sorted(glob.glob(args.states, recursive=True))]
    if args.limit:
        paths = paths[: args.limit]
    experts = [value.strip() for value in args.experts.split(",") if value.strip()]
    levels = [value.strip() for value in args.levels.split(",") if value.strip()]
    invalid_experts = sorted(set(experts) - set(EXPERTS))
    invalid_levels = sorted(set(levels) - set(LEVELS))
    if invalid_experts or invalid_levels:
        raise SystemExit(
            f"invalid experts={invalid_experts}, levels={invalid_levels}"
        )
    summary = collect(
        paths,
        out,
        episodes_per_state=args.episodes_per_state,
        experts=experts,
        levels=levels,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_steps=args.max_steps,
        max_actions=args.max_actions,
        image_size=args.image_size,
        shard_size=args.shard_size,
        forbidden=forbidden_views(Path(args.holdout), Path(args.human_gold_dir)),
        beam_width=args.beam_width,
        beam_depth=args.beam_depth,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["view_overlap"]:
        raise SystemExit("train/val view leakage detected")


if __name__ == "__main__":
    main()
