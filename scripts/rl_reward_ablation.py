#!/usr/bin/env python3
"""Run formula-only vs small-beta human-reward PPO ablations."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.training.human_reward import (  # noqa: E402
    require_human_reward_gate,
)
from layout_agent.training.train_ppo import (  # noqa: E402
    _forbidden_views,
    train_ppo,
)


def _tag(beta: float) -> str:
    return "formula_only" if beta == 0 else f"human_beta_{beta:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument("--bc-gate", required=True)
    parser.add_argument("--reward-checkpoint", required=True)
    parser.add_argument("--reward-gate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--betas", default="0,0.01,0.03")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument(
        "--holdout",
        default=str(ROOT / "data" / "evalsys" / "holdout_views.txt"),
    )
    parser.add_argument(
        "--human-gold-dir",
        default=str(ROOT / "data" / "evalsys" / "human_gold"),
    )
    args = parser.parse_args()
    betas = [float(value.strip()) for value in args.betas.split(",") if value.strip()]
    if not betas or any(beta < 0 for beta in betas):
        parser.error("--betas must contain non-negative numbers")
    if any(beta > 0 for beta in betas):
        require_human_reward_gate(
            Path(args.reward_gate),
            Path(args.reward_checkpoint),
        )
    states = [Path(value) for value in sorted(glob.glob(args.states, recursive=True))]
    forbidden = _forbidden_views(
        Path(args.holdout),
        Path(args.human_gold_dir),
    )
    out = Path(args.out)
    results: List[Dict[str, Any]] = []
    for beta in betas:
        tag = _tag(beta)
        env_extra = None
        if beta > 0:
            env_extra = {
                "human_reward_checkpoint": Path(args.reward_checkpoint),
                "human_reward_gate": Path(args.reward_gate),
                "human_reward_beta": beta,
                "human_reward_device": args.device or None,
            }
        report = train_ppo(
            states,
            Path(args.bc_checkpoint),
            Path(args.bc_gate),
            out / tag,
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            max_steps=args.max_steps,
            max_actions=args.max_actions,
            image_size=args.image_size,
            seed=args.seed,
            device_name=args.device,
            subprocess=not args.sync,
            forbidden_view_keys=forbidden,
            env_extra=env_extra,
        )
        results.append(
            {
                "tag": tag,
                "beta": beta,
                "best_score": report["best_score"],
                "global_step": report["global_step"],
                "action_coverage": report["action_coverage"],
                "report": str((out / tag / "report.json").as_posix()),
            }
        )
    summary = {
        "schema_version": 1,
        "reward_gate": str(Path(args.reward_gate).as_posix()),
        "same_seed_and_budget": True,
        "runs": results,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
