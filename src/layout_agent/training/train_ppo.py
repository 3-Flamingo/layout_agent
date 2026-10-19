"""Action-masked PPO initialized from a behavior-cloning checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from layout_agent.evalset.isolation import forbidden_training_views
from layout_agent.training.policy import (
    CandidatePolicyNet,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from layout_agent.training.vector_env import (
    SubprocessLayoutVectorEnv,
    SyncLayoutVectorEnv,
)


def _validate_bc_gate(path: Path, checkpoint: Path) -> Dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not bool((report.get("gate") or {}).get("passed")):
        raise ValueError(f"BC rollout gate did not pass: {path}")
    recorded = str(report.get("checkpoint") or "")
    if recorded and Path(recorded).resolve() != checkpoint.resolve():
        raise ValueError(
            f"BC gate checkpoint mismatch: gate={recorded}, requested={checkpoint}"
        )
    return report


def _torch_observation(
    observation: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: (
            torch.as_tensor(value, device=device).bool()
            if key == "action_mask"
            else torch.as_tensor(value, device=device).float()
        )
        for key, value in observation.items()
    }


def _compress_observation(
    observation: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    return {
        "image": np.clip(observation["image"] * 255.0, 0, 255).astype(np.uint8),
        "global_features": observation["global_features"].astype(np.float16),
        "action_features": observation["action_features"].astype(np.float16),
        "action_mask": observation["action_mask"].astype(np.bool_),
    }


def _flatten_observations(
    rows: Sequence[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    stacked = {
        key: np.stack([row[key] for row in rows])
        for key in rows[0]
    }
    time_steps, num_envs = stacked["image"].shape[:2]
    flattened: Dict[str, np.ndarray] = {}
    for key, value in stacked.items():
        flattened[key] = value.reshape(
            (time_steps * num_envs, *value.shape[2:])
        )
    return flattened


def _batch_observation(
    flattened: Dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in flattened.items():
        tensor = torch.as_tensor(value[indices], device=device)
        if key == "action_mask":
            out[key] = tensor.bool()
        elif key == "image":
            out[key] = tensor.float() / 255.0
        else:
            out[key] = tensor.float()
    return out


def _gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros((rewards.shape[1],), dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        next_values = last_values if step == rewards.shape[0] - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step].astype(np.float32)
        delta = rewards[step] + gamma * next_values * nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[step] = last_gae
    return advantages, advantages + values


def train_ppo(
    state_paths: Sequence[Path],
    bc_checkpoint: Path,
    bc_gate: Path,
    out: Path,
    *,
    total_timesteps: int = 100_000,
    num_envs: int = 8,
    rollout_steps: int = 128,
    update_epochs: int = 4,
    minibatch_size: int = 256,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.03,
    max_steps: int = 12,
    max_actions: int = 512,
    image_size: int = 256,
    seed: int = 17,
    device_name: str = "",
    subprocess: bool = True,
    forbidden_view_keys: Sequence[str] = (),
    env_extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not state_paths:
        raise ValueError("state_paths cannot be empty")
    gate_report = _validate_bc_gate(bc_gate, bc_checkpoint)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, checkpoint = load_policy_checkpoint(bc_checkpoint, device=device)
    if checkpoint.get("kind") != "behavior_cloning":
        raise ValueError("PPO must initialize from a behavior_cloning checkpoint")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    out.mkdir(parents=True, exist_ok=True)
    env_kwargs: Dict[str, Any] = {
        "max_steps": max_steps,
        "img_size": image_size,
        "max_actions": max_actions,
        "forbidden_view_keys": list(forbidden_view_keys),
        "work_dir": out / "_env_work",
        **(env_extra or {}),
    }
    vector_class = (
        SubprocessLayoutVectorEnv
        if subprocess and num_envs > 1
        else SyncLayoutVectorEnv
    )
    vector = vector_class(
        state_paths,
        num_envs,
        env_kwargs,
        seed=seed,
    )
    observation, _ = vector.reset(
        [seed + index for index in range(num_envs)]
    )
    updates = max(
        1,
        math.ceil(total_timesteps / max(rollout_steps * num_envs, 1)),
    )
    global_step = 0
    history: List[Dict[str, Any]] = []
    action_coverage: Counter[str] = Counter()
    best_score = -float("inf")
    started = time.perf_counter()
    try:
        for update in range(1, updates + 1):
            observations: List[Dict[str, np.ndarray]] = []
            actions: List[np.ndarray] = []
            log_probs: List[np.ndarray] = []
            values: List[np.ndarray] = []
            rewards: List[np.ndarray] = []
            dones: List[np.ndarray] = []
            episode_rows: List[Dict[str, Any]] = []
            step_times: List[float] = []
            no_ops = rollbacks = 0
            for _ in range(rollout_steps):
                observations.append(_compress_observation(observation))
                with torch.no_grad():
                    distribution, value = model.distribution(
                        _torch_observation(observation, device)
                    )
                    action = distribution.sample()
                    log_prob = distribution.log_prob(action)
                next_observation, reward, terminated, truncated, infos = vector.step(
                    action.cpu().numpy()
                )
                done = np.logical_or(terminated, truncated)
                actions.append(action.cpu().numpy())
                log_probs.append(log_prob.cpu().numpy())
                values.append(value.cpu().numpy())
                rewards.append(reward)
                dones.append(done)
                for info in infos:
                    action_info = info.get("action") or {}
                    if action_info.get("type"):
                        action_coverage[str(action_info["type"])] += 1
                    no_ops += int(bool(info.get("no_op")))
                    rollbacks += int(bool(info.get("rolled_back")))
                    step_times.append(float(info.get("step_time_ms", 0.0)))
                    if "final_info" in info:
                        episode_rows.append(info["final_info"])
                observation = next_observation
                global_step += num_envs
            with torch.no_grad():
                _, last_values_tensor = model(
                    **_torch_observation(observation, device)
                )
            reward_array = np.asarray(rewards, dtype=np.float32)
            value_array = np.asarray(values, dtype=np.float32)
            done_array = np.asarray(dones, dtype=np.bool_)
            advantages, returns = _gae(
                reward_array,
                value_array,
                done_array,
                last_values_tensor.cpu().numpy(),
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            flat_obs = _flatten_observations(observations)
            flat_actions = np.asarray(actions).reshape(-1)
            flat_log_probs = np.asarray(log_probs, dtype=np.float32).reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_values = value_array.reshape(-1)
            flat_advantages = (
                flat_advantages - flat_advantages.mean()
            ) / (flat_advantages.std() + 1e-8)
            batch_size = len(flat_actions)
            indices = np.arange(batch_size)
            losses: List[Dict[str, float]] = []
            stop_for_kl = False
            for _ in range(update_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, minibatch_size):
                    mb = indices[start : start + minibatch_size]
                    batch_observation = _batch_observation(flat_obs, mb, device)
                    batch_actions = torch.as_tensor(
                        flat_actions[mb],
                        device=device,
                        dtype=torch.long,
                    )
                    new_log_prob, entropy, new_value = model.evaluate_actions(
                        batch_observation,
                        batch_actions,
                    )
                    old_log_prob = torch.as_tensor(
                        flat_log_probs[mb],
                        device=device,
                    )
                    ratio = (new_log_prob - old_log_prob).exp()
                    batch_advantage = torch.as_tensor(
                        flat_advantages[mb],
                        device=device,
                    )
                    policy_loss = torch.max(
                        -batch_advantage * ratio,
                        -batch_advantage
                        * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
                    ).mean()
                    target_value = torch.as_tensor(
                        flat_returns[mb],
                        device=device,
                    )
                    old_value = torch.as_tensor(
                        flat_values[mb],
                        device=device,
                    )
                    clipped_value = old_value + torch.clamp(
                        new_value - old_value,
                        -clip_coef,
                        clip_coef,
                    )
                    value_loss = 0.5 * torch.max(
                        (new_value - target_value).pow(2),
                        (clipped_value - target_value).pow(2),
                    ).mean()
                    entropy_loss = entropy.mean()
                    loss = (
                        policy_loss
                        + value_coef * value_loss
                        - entropy_coef * entropy_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_grad_norm,
                    )
                    optimizer.step()
                    approximate_kl = (
                        (ratio - 1.0) - (new_log_prob - old_log_prob)
                    ).mean()
                    losses.append(
                        {
                            "policy": float(policy_loss.detach().item()),
                            "value": float(value_loss.detach().item()),
                            "entropy": float(entropy_loss.detach().item()),
                            "kl": float(approximate_kl.detach().item()),
                        }
                    )
                    if target_kl > 0 and approximate_kl.item() > target_kl:
                        stop_for_kl = True
                        break
                if stop_for_kl:
                    break
            completed_returns = [
                float(row.get("P", 0.0)) for row in episode_rows
            ]
            hard_clear = [
                not bool(row.get("hard_fail", True)) for row in episode_rows
            ]
            row = {
                "update": update,
                "global_step": global_step,
                "episodes_completed": len(episode_rows),
                "hard_clear_rate": float(np.mean(hard_clear))
                if hard_clear
                else None,
                "mean_final_P": float(np.mean(completed_returns))
                if completed_returns
                else None,
                "mean_rollout_reward": float(reward_array.mean()),
                "no_op_rate": no_ops / max(rollout_steps * num_envs, 1),
                "rollback_rate": rollbacks / max(rollout_steps * num_envs, 1),
                "mean_step_time_ms": float(np.mean(step_times))
                if step_times
                else 0.0,
                "loss": {
                    key: float(np.mean([item[key] for item in losses]))
                    for key in ("policy", "value", "entropy", "kl")
                },
            }
            history.append(row)
            score = (
                float(row["hard_clear_rate"])
                if row["hard_clear_rate"] is not None
                else float(row["mean_rollout_reward"])
            )
            print(
                f"update {update:04d} step={global_step} "
                f"episodes={row['episodes_completed']} score={score:.4f} "
                f"reward={row['mean_rollout_reward']:.4f}"
            )
            if score > best_score:
                best_score = score
                save_policy_checkpoint(
                    out / "best.pt",
                    model,
                    kind="ppo",
                    step=global_step,
                    metrics=row,
                    optimizer=optimizer,
                    extra={
                        "bc_checkpoint": str(bc_checkpoint),
                        "bc_gate": str(bc_gate),
                    },
                )
    finally:
        vector.close()
    save_policy_checkpoint(
        out / "last.pt",
        model,
        kind="ppo",
        step=global_step,
        metrics=history[-1],
        optimizer=optimizer,
        extra={"bc_checkpoint": str(bc_checkpoint), "bc_gate": str(bc_gate)},
    )
    report = {
        "kind": "masked_ppo",
        "initialized_from": str(bc_checkpoint.as_posix()),
        "bc_gate": str(bc_gate.as_posix()),
        "bc_gate_summary": gate_report.get("gate"),
        "device": str(device),
        "vectorization": vector_class.__name__,
        "num_envs": num_envs,
        "global_step": global_step,
        "updates": updates,
        "config": {
            "total_timesteps": total_timesteps,
            "rollout_steps": rollout_steps,
            "update_epochs": update_epochs,
            "minibatch_size": minibatch_size,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_coef": clip_coef,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
            "max_grad_norm": max_grad_norm,
            "target_kl": target_kl,
            "max_steps": max_steps,
            "max_actions": max_actions,
            "image_size": image_size,
            "seed": seed,
        },
        "action_coverage": dict(action_coverage),
        "best_score": best_score,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _forbidden_views(path: Path, human_gold_dir: Path) -> List[str]:
    root = Path(__file__).resolve().parents[3]
    return sorted(
        forbidden_training_views(
            root,
            holdout=path,
            gold_dir=human_gold_dir,
        )
    )


def main() -> None:
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument("--bc-gate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--human-reward-checkpoint", default="")
    parser.add_argument("--human-reward-gate", default="")
    parser.add_argument("--human-reward-beta", type=float, default=0.0)
    parser.add_argument(
        "--holdout",
        default=str(
            Path(__file__).resolve().parents[3]
            / "data"
            / "evalsys"
            / "holdout_views.txt"
        ),
    )
    parser.add_argument(
        "--human-gold-dir",
        default=str(
            Path(__file__).resolve().parents[3]
            / "data"
            / "evalsys"
            / "human_gold"
        ),
    )
    args = parser.parse_args()
    if args.human_reward_beta > 0 and not (
        args.human_reward_checkpoint and args.human_reward_gate
    ):
        parser.error(
            "--human-reward-beta > 0 requires --human-reward-checkpoint "
            "and --human-reward-gate"
        )
    report = train_ppo(
        [Path(value) for value in sorted(glob.glob(args.states, recursive=True))],
        Path(args.bc_checkpoint),
        Path(args.bc_gate),
        Path(args.out),
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        max_steps=args.max_steps,
        max_actions=args.max_actions,
        image_size=args.image_size,
        seed=args.seed,
        device_name=args.device,
        subprocess=not args.sync,
        forbidden_view_keys=_forbidden_views(
            Path(args.holdout),
            Path(args.human_gold_dir),
        ),
        env_extra=(
            {
                "human_reward_checkpoint": Path(args.human_reward_checkpoint),
                "human_reward_gate": Path(args.human_reward_gate),
                "human_reward_beta": args.human_reward_beta,
                "human_reward_device": args.device or None,
            }
            if args.human_reward_beta > 0
            else None
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
