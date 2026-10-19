from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from layout_agent.models import EvalReport
from layout_agent.training.human_reward import (
    GatedHumanReward,
    build_human_reward_gate,
    require_human_reward_gate,
)


class _Scorer:
    def __init__(self, before: float, after: float):
        self.values = iter((before, after))

    def score_state(self, _state: object) -> float:
        return next(self.values)


class HumanRewardTest(unittest.TestCase):
    def test_gate_requires_counts_and_both_metric_improvements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "data" / "evalsys" / "human_gold"
            gold.mkdir(parents=True)
            t1 = gold / "t1_labels.jsonl"
            t2 = gold / "t2_labels.jsonl"
            t1.write_text("{}\n", encoding="utf-8")
            t2.write_text("{}\n", encoding="utf-8")

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            manifest = gold / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "human_labels_frozen",
                        "labels": {
                            "t1": "data/evalsys/human_gold/t1_labels.jsonl",
                            "t2": "data/evalsys/human_gold/t2_labels.jsonl",
                        },
                    }
                ),
                encoding="utf-8",
            )
            model = {
                "inputs": {
                    "t1": {"sha256": digest(t1)},
                    "t2": {"sha256": digest(t2)},
                },
                "t1": {"n": 15, "spearman": 0.6},
                "t2": {"overall": {"n": 80, "strict_accuracy": 0.75}},
            }
            formula = {
                "inputs": {
                    "t1": {"sha256": digest(t1)},
                    "t2": {"sha256": digest(t2)},
                },
                "t1": {"n": 15, "spearman": 0.5},
                "t2": {"overall": {"n": 80, "strict_accuracy": 0.65}},
            }
            checkpoint = root / "reward.pt"
            checkpoint.write_bytes(b"checkpoint")
            model["ckpt"] = str(checkpoint.resolve())
            model["checkpoint_sha256"] = digest(checkpoint)
            gate = build_human_reward_gate(
                model,
                formula,
                reward_checkpoint=checkpoint,
                gold_manifest=manifest,
            )
            self.assertTrue(gate["passed"])
            self.assertTrue(all(gate["checks"].values()))
            other_checkpoint = root / "other.pt"
            other_checkpoint.write_bytes(b"other")
            substituted = build_human_reward_gate(
                model,
                formula,
                reward_checkpoint=other_checkpoint,
                gold_manifest=manifest,
            )
            self.assertFalse(substituted["passed"])
            self.assertFalse(substituted["checks"]["model_checkpoint_matches"])
            model["t2"]["overall"]["n"] = 79
            failed = build_human_reward_gate(
                model,
                formula,
                reward_checkpoint=checkpoint,
                gold_manifest=manifest,
            )
            self.assertFalse(failed["passed"])
            self.assertFalse(failed["checks"]["t2_count"])
            with self.assertRaises(ValueError):
                build_human_reward_gate(
                    model,
                    formula,
                    reward_checkpoint=checkpoint,
                    gold_manifest=manifest,
                    min_t1=1,
                    min_t2=1,
                )

    def test_learned_delta_only_applies_when_both_states_are_hard_clean(self) -> None:
        clean = EvalReport(state_id="s", hard_fail=False)
        hard = EvalReport(state_id="s", hard_fail=True)
        reward = GatedHumanReward(  # type: ignore[arg-type]
            scorer=_Scorer(40.0, 55.0),
            beta=0.1,
            gate={"passed": True},
        )
        value, detail = reward.delta(object(), object(), clean, clean)  # type: ignore[arg-type]
        self.assertAlmostEqual(value, 0.015)
        self.assertTrue(detail["human_reward_applied"])
        blocked = GatedHumanReward(  # type: ignore[arg-type]
            scorer=_Scorer(40.0, 90.0),
            beta=0.1,
            gate={"passed": True},
        )
        value, detail = blocked.delta(object(), object(), hard, clean)  # type: ignore[arg-type]
        self.assertEqual(value, 0.0)
        self.assertFalse(detail["human_reward_applied"])

    def test_runtime_gate_fails_closed_when_provenance_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "reward.pt"
            checkpoint.write_bytes(b"not-a-real-checkpoint")
            gate = root / "gate.json"
            gate.write_text(json.dumps({"passed": True}), encoding="utf-8")
            with self.assertRaises(ValueError):
                GatedHumanReward.load(
                    checkpoint,
                    gate,
                    beta=0.01,
                    device="cpu",
                )

    def test_awaiting_labels_cannot_pass_the_human_reward_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = root / "data" / "evalsys" / "human_gold"
            gold.mkdir(parents=True)
            manifest = gold / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "tasks_frozen; awaiting human labels",
                        "labels": {"t1": None, "t2": None},
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = root / "reward.pt"
            checkpoint.write_bytes(b"checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            model = {
                "ckpt": str(checkpoint.resolve()),
                "checkpoint_sha256": digest,
                "inputs": {"t1": {"sha256": "a"}, "t2": {"sha256": "b"}},
                "t1": {"n": 15, "spearman": 0.9},
                "t2": {"overall": {"n": 80, "strict_accuracy": 0.9}},
            }
            formula = {
                "inputs": {"t1": {"sha256": "a"}, "t2": {"sha256": "b"}},
                "t1": {"n": 15, "spearman": 0.1},
                "t2": {"overall": {"n": 80, "strict_accuracy": 0.1}},
            }
            gate = build_human_reward_gate(
                model,
                formula,
                reward_checkpoint=checkpoint,
                gold_manifest=manifest,
            )
            self.assertFalse(gate["passed"])
            self.assertFalse(gate["checks"]["gold_manifest_frozen"])
            (root / "gate.json").write_text(
                json.dumps(gate),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                require_human_reward_gate(root / "gate.json", checkpoint)


if __name__ == "__main__":
    unittest.main()
