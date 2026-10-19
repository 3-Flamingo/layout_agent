from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch

from layout_agent.training.policy import (
    CandidatePolicyNet,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from layout_agent.training.rl_features import ACTION_FEATURE_DIM, GLOBAL_FEATURE_DIM
from layout_agent.training.trajectory import (
    TrajectoryDataset,
    TrajectoryShardWriter,
    summarize_trajectories,
)
from layout_agent.training.train_ppo import _gae, _validate_bc_gate
from layout_agent.training.collect import collect, forbidden_views


def _observation(mask: list[int]) -> dict[str, np.ndarray]:
    actions = len(mask)
    return {
        "image": np.zeros((3, 16, 16), dtype=np.float32),
        "global_features": np.zeros((GLOBAL_FEATURE_DIM,), dtype=np.float32),
        "action_features": np.zeros(
            (actions, ACTION_FEATURE_DIM),
            dtype=np.float32,
        ),
        "action_mask": np.asarray(mask, dtype=np.int8),
    }


class RlTrainingTest(unittest.TestCase):
    def test_candidate_policy_never_selects_masked_action(self) -> None:
        model = CandidatePolicyNet(hidden_dim=16)
        observation = _observation([1, 1, 0, 0])
        batch = {
            key: torch.from_numpy(value).unsqueeze(0)
            for key, value in observation.items()
        }
        logits, value = model(
            batch["image"],
            batch["global_features"],
            batch["action_features"],
            batch["action_mask"],
        )
        self.assertEqual(tuple(logits.shape), (1, 4))
        self.assertEqual(tuple(value.shape), (1,))
        self.assertLess(float(logits[0, 2].detach()), -1e20)
        action, _, _, _ = model.act(batch, deterministic=True)
        self.assertIn(int(action.item()), {0, 1})

    def test_trajectory_shards_preserve_view_split_and_expert_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with TrajectoryShardWriter(root, shard_size=1) as writer:
                writer.add(
                    _observation([1, 1, 0, 0]),
                    chosen_action=1,
                    reward=1.0,
                    done=False,
                    split="train",
                    expert="rules",
                    level="heavy",
                    episode_id="e1",
                    view_key="train-view",
                )
                writer.add(
                    _observation([1, 0, 0, 0]),
                    chosen_action=0,
                    reward=2.0,
                    done=True,
                    split="val",
                    expert="random",
                    level="clean",
                    episode_id="e2",
                    view_key="val-view",
                )
            train = TrajectoryDataset(root, split="train", experts=["rules"])
            val_expert = TrajectoryDataset(root, split="val", experts=["rules"])
            self.assertEqual(len(train), 1)
            self.assertEqual(len(val_expert), 0)
            self.assertEqual(train[0]["chosen_action"], 1)
            train.close()
            val_expert.close()
            summary = summarize_trajectories(root)
            self.assertEqual(summary["transitions"], 2)
            self.assertEqual(summary["view_overlap"], [])

    def test_policy_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            model = CandidatePolicyNet(hidden_dim=16)
            save_policy_checkpoint(
                path,
                model,
                kind="behavior_cloning",
                step=3,
                metrics={"top1": 0.5},
            )
            loaded, metadata = load_policy_checkpoint(path)
            self.assertEqual(metadata["kind"], "behavior_cloning")
            self.assertEqual(loaded.config["hidden_dim"], 16)
            for before, after in zip(model.parameters(), loaded.parameters()):
                self.assertTrue(torch.equal(before, after))

    def test_gae_stops_bootstrap_at_episode_boundary(self) -> None:
        rewards = np.asarray([[1.0], [1.0]], dtype=np.float32)
        values = np.zeros_like(rewards)
        dones = np.asarray([[False], [True]])
        advantages, returns = _gae(
            rewards,
            values,
            dones,
            np.asarray([10.0], dtype=np.float32),
            gamma=1.0,
            gae_lambda=1.0,
        )
        np.testing.assert_allclose(advantages[:, 0], [2.0, 1.0])
        np.testing.assert_allclose(returns, advantages)

    def test_ppo_rejects_failed_or_mismatched_bc_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "bc.pt"
            checkpoint.write_bytes(b"x")
            gate = root / "gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "gate": {"passed": False},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _validate_bc_gate(gate, checkpoint)
            other = root / "other.pt"
            other.write_bytes(b"y")
            gate.write_text(
                json.dumps(
                    {
                        "checkpoint": str(other),
                        "gate": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _validate_bc_gate(gate, checkpoint)

    def test_forbidden_views_include_gold_and_colleague(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout.txt"
            holdout.write_text("hold/v1\n", encoding="utf-8")
            gold = root / "human_gold"
            gold.mkdir()
            (gold / "t1_tasks.jsonl").write_text(
                json.dumps({"id": "t1", "view_key": "gold/t1"}) + "\n",
                encoding="utf-8",
            )
            (gold / "t2_tasks.jsonl").write_text(
                json.dumps({"id": "t2", "view_key": "gold/t2"}) + "\n",
                encoding="utf-8",
            )
            challenge = root / "data" / "evalsys" / "proxy"
            challenge.mkdir(parents=True)
            (challenge / "colleague_challenge.jsonl").write_text(
                json.dumps({"view_key": "col/c1"}) + "\n",
                encoding="utf-8",
            )
            keys = forbidden_views(holdout, gold, root=root)
            self.assertEqual(keys, {"hold/v1", "gold/t1", "gold/t2", "col/c1"})


if __name__ == "__main__":
    unittest.main()
