"""Compressed trajectory shards for behavior cloning and PPO warm starts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class TrajectoryShardWriter:
    def __init__(self, root: Path, shard_size: int = 256):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = max(int(shard_size), 1)
        self._rows: List[Dict[str, Any]] = []
        self._shard = len(list(self.root.glob("shard_*.npz")))
        self._meta = (self.root / "transitions.jsonl").open("a", encoding="utf-8")
        self.count = 0

    def add(
        self,
        observation: Dict[str, np.ndarray],
        *,
        chosen_action: int,
        reward: float,
        done: bool,
        split: str,
        expert: str,
        level: str,
        episode_id: str,
        view_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        image = np.clip(observation["image"] * 255.0, 0, 255).astype(np.uint8)
        row = {
            "image": image,
            "global_features": observation["global_features"].astype(np.float32),
            "action_features": observation["action_features"].astype(np.float16),
            "action_mask": observation["action_mask"].astype(np.uint8),
            "chosen_action": int(chosen_action),
            "reward": float(reward),
            "done": bool(done),
            "split": 1 if split == "val" else 0,
            "expert": str(expert),
            "level": str(level),
            "episode_id": str(episode_id),
            "view_key": str(view_key),
        }
        if not row["action_mask"][row["chosen_action"]]:
            raise ValueError("chosen action is masked")
        self._rows.append(row)
        record = {
            "sample_index": self.count,
            "chosen_action": int(chosen_action),
            "reward": float(reward),
            "done": bool(done),
            "split": split,
            "expert": expert,
            "level": level,
            "episode_id": episode_id,
            "view_key": view_key,
            **(metadata or {}),
        }
        self._meta.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1
        if len(self._rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        path = self.root / f"shard_{self._shard:05d}.npz"
        keys = self._rows[0].keys()
        arrays = {
            key: np.asarray([row[key] for row in self._rows])
            for key in keys
        }
        np.savez_compressed(path, **arrays)
        self._rows.clear()
        self._shard += 1
        self._meta.flush()

    def close(self) -> None:
        self.flush()
        self._meta.close()

    def __enter__(self) -> "TrajectoryShardWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class TrajectoryDataset:
    """Map-style torch-compatible dataset backed by compressed NPZ shards."""

    def __init__(
        self,
        root: Path,
        *,
        split: str = "",
        experts: Optional[Sequence[str]] = None,
    ):
        self.root = Path(root)
        self.paths = sorted(self.root.glob("shard_*.npz"))
        if not self.paths:
            raise ValueError(f"no trajectory shards under {self.root}")
        self._shards = [np.load(path, allow_pickle=False) for path in self.paths]
        expert_set = set(experts or ())
        wanted_split = None if not split else (1 if split == "val" else 0)
        self.index: List[Tuple[int, int]] = []
        for shard_index, shard in enumerate(self._shards):
            for row_index in range(len(shard["chosen_action"])):
                if wanted_split is not None and int(shard["split"][row_index]) != wanted_split:
                    continue
                if expert_set and str(shard["expert"][row_index]) not in expert_set:
                    continue
                self.index.append((shard_index, row_index))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        shard_index, row_index = self.index[index]
        shard = self._shards[shard_index]
        return {
            "image": shard["image"][row_index].astype(np.float32) / 255.0,
            "global_features": shard["global_features"][row_index].astype(np.float32),
            "action_features": shard["action_features"][row_index].astype(np.float32),
            "action_mask": shard["action_mask"][row_index].astype(np.bool_),
            "chosen_action": int(shard["chosen_action"][row_index]),
            "reward": float(shard["reward"][row_index]),
            "done": bool(shard["done"][row_index]),
            "split": "val" if int(shard["split"][row_index]) else "train",
            "expert": str(shard["expert"][row_index]),
            "level": str(shard["level"][row_index]),
            "episode_id": str(shard["episode_id"][row_index]),
            "view_key": str(shard["view_key"][row_index]),
        }

    def close(self) -> None:
        for shard in self._shards:
            shard.close()


def summarize_trajectories(root: Path) -> Dict[str, Any]:
    dataset = TrajectoryDataset(root)
    split = Counter()
    experts = Counter()
    levels = Counter()
    views: Dict[str, set[str]] = {"train": set(), "val": set()}
    episodes: set[str] = set()
    for index in range(len(dataset)):
        row = dataset[index]
        split[row["split"]] += 1
        experts[row["expert"]] += 1
        levels[row["level"]] += 1
        views[row["split"]].add(row["view_key"])
        episodes.add(row["episode_id"])
    dataset.close()
    overlap = sorted(views["train"] & views["val"])
    return {
        "transitions": sum(split.values()),
        "episodes": len(episodes),
        "split_transitions": dict(split),
        "expert_transitions": dict(experts),
        "level_transitions": dict(levels),
        "train_views": len(views["train"]),
        "val_views": len(views["val"]),
        "view_overlap": overlap,
    }
