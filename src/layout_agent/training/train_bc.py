"""Train the shared candidate-action network by behavior cloning."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from layout_agent.training.policy import CandidatePolicyNet, save_policy_checkpoint
from layout_agent.training.trajectory import TrajectoryDataset


def _collate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    observation = {
        key: torch.from_numpy(np.stack([row[key] for row in rows]))
        for key in (
            "image",
            "global_features",
            "action_features",
            "action_mask",
        )
    }
    return {
        "observation": observation,
        "chosen_action": torch.tensor(
            [row["chosen_action"] for row in rows],
            dtype=torch.long,
        ),
        "expert": [row["expert"] for row in rows],
    }


def _to_device(
    observation: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: (
            value.to(device, non_blocking=True).bool()
            if key == "action_mask"
            else value.to(device, non_blocking=True).float()
        )
        for key, value in observation.items()
    }


@torch.no_grad()
def validate(
    model: CandidatePolicyNet,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    correct = defaultdict(int)
    total = defaultdict(int)
    losses: List[float] = []
    for batch in loader:
        observation = _to_device(batch["observation"], device)
        targets = batch["chosen_action"].to(device)
        logits, _ = model(**observation)
        losses.append(float(F.cross_entropy(logits, targets).item()))
        predictions = logits.argmax(dim=-1)
        matches = (predictions == targets).cpu().tolist()
        for matched, expert in zip(matches, batch["expert"]):
            total["all"] += 1
            total[expert] += 1
            if matched:
                correct["all"] += 1
                correct[expert] += 1
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "top1": {
            key: correct[key] / total[key] if total[key] else 0.0
            for key in sorted(total)
        },
        "n": dict(total),
    }


def train(
    trajectories: Path,
    out: Path,
    *,
    experts: Sequence[str],
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    patience: int = 5,
    seed: int = 17,
    device_name: str = "",
    num_workers: int = 0,
    min_val_accuracy: float = 0.0,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_data = TrajectoryDataset(
        trajectories,
        split="train",
        experts=experts,
    )
    val_data = TrajectoryDataset(
        trajectories,
        split="val",
        experts=experts,
    )
    if not len(train_data) or not len(val_data):
        raise ValueError(
            f"BC needs view-disjoint train and val samples; "
            f"got train={len(train_data)}, val={len(val_data)}"
        )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    first = train_data[0]
    model = CandidatePolicyNet(
        in_channels=int(first["image"].shape[0]),
        global_dim=int(first["global_features"].shape[0]),
        action_dim=int(first["action_features"].shape[1]),
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    out.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    best = -1.0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses: List[float] = []
        for batch in train_loader:
            observation = _to_device(batch["observation"], device)
            targets = batch["chosen_action"].to(device)
            logits, _ = model(**observation)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        metrics = validate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val": metrics,
        }
        history.append(row)
        score = float(metrics["top1"].get("all", 0.0))
        print(
            f"epoch {epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f} val_top1={score:.4f}"
        )
        if score > best + 1e-9:
            best = score
            stale = 0
            save_policy_checkpoint(
                out / "best.pt",
                model,
                kind="behavior_cloning",
                step=epoch,
                metrics=metrics,
                optimizer=optimizer,
                extra={
                    "experts": list(experts),
                    "trajectories": str(trajectories),
                    "seed": seed,
                },
            )
        else:
            stale += 1
            if stale >= patience:
                break
    report = {
        "kind": "behavior_cloning",
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "experts": list(experts),
        "device": str(device),
        "epochs_completed": len(history),
        "best_val_top1": best,
        "min_val_accuracy": min_val_accuracy,
        "offline_gate_passed": best >= min_val_accuracy,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "checkpoint": str((out / "best.pt").as_posix()),
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    train_data.close()
    val_data.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--experts", default="rules,greedy,beam")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-val-accuracy", type=float, default=0.0)
    args = parser.parse_args()
    report = train(
        Path(args.trajectories),
        Path(args.out),
        experts=[value.strip() for value in args.experts.split(",") if value.strip()],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        patience=args.patience,
        seed=args.seed,
        device_name=args.device,
        num_workers=args.num_workers,
        min_val_accuracy=args.min_val_accuracy,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["offline_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
