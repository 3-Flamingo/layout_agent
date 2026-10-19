from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from layout_agent.evalset.pairs import split_for
from layout_agent.training.collect import collect
from layout_agent.training.env import LayoutEnv
from layout_agent.training.expert import ExpertPolicy
from layout_agent.training.train_bc import train as train_bc
from layout_agent.training.train_ppo import train_ppo


def _state_dict(view_uuid: str) -> dict:
    return {
        "state_id": f"pipe_{view_uuid}",
        "source": {"solution_id": "pipe", "view_uuid": view_uuid},
        "view": {
            "bbox": {
                "min": {"x": 0.0, "y": 0.0},
                "max": {"x": 100.0, "y": 80.0},
            }
        },
        "raster": {
            "instance_image": "missing.png",
            "obstacle_image": None,
            "width": 8,
            "height": 8,
            "origin": {"x": 0.0, "y": 80.0},
            "pixel_size": 1.0,
        },
        "color_legend": {
            "entries": {
                "ff0000": {"id": "lin", "category": "linear"},
                "0000ff": {"id": "lin2", "category": "linear"},
                "00ff00": {"id": "dia", "category": "diameter"},
            }
        },
        "linears": [
            {
                "id": "lin",
                "color": "ff0000",
                "p1": {"x": 20.0, "y": 20.0},
                "p2": {"x": 80.0, "y": 20.0},
                "dir": {"x": 0.0, "y": 1.0},
                "point": {"x": 50.0, "y": 32.0},
                "value": 60.0,
                "text": {"content": "60", "height": 2.5},
            },
            {
                "id": "lin2",
                "color": "0000ff",
                "p1": {"x": 30.0, "y": 40.0},
                "p2": {"x": 70.0, "y": 40.0},
                "dir": {"x": 0.0, "y": 1.0},
                "point": {"x": 50.0, "y": 48.0},
                "value": 40.0,
                "text": {"content": "40", "height": 2.5},
            },
        ],
        "diameters": [
            {
                "id": "dia",
                "category": "diameter",
                "color": "00ff00",
                "anchor": {"x": 50.0, "y": 50.0},
                "path": [{"x": 50.0, "y": 50.0}, {"x": 72.0, "y": 50.0}],
                "text": {"content": "⌀10", "height": 2.5},
            }
        ],
    }


class RlPipelineTest(unittest.TestCase):
    def test_experts_cover_current_action_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "fixture.state.json"
            state_path.write_text(json.dumps(_state_dict("exp")), encoding="utf-8")
            env = LayoutEnv(
                [state_path],
                img_size=32,
                max_actions=64,
                max_steps=3,
                work_dir=root / "env",
                seed=7,
            )
            env.reset(seed=7, options={"index": 0})
            kinds = set()
            for name in ("rules", "greedy", "beam", "random"):
                policy = ExpertPolicy(name, seed=7, beam_width=2, beam_depth=1)
                index = policy.choose(env)
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, len(env.legal_actions))
                kinds.add(env.legal_actions[index].type)
            self.assertTrue(kinds)

    def test_collect_bc_and_masked_ppo_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            states_dir = root / "states"
            states_dir.mkdir()
            val_ratio = 0.4
            train_paths: list[Path] = []
            val_paths: list[Path] = []
            index = 0
            while len(train_paths) < 2 or len(val_paths) < 2:
                index += 1
                view = f"v{index:03d}"
                path = states_dir / f"{view}.state.json"
                path.write_text(json.dumps(_state_dict(view)), encoding="utf-8")
                bucket = train_paths if split_for(f"pipe/{view}", val_ratio) == "train" else val_paths
                bucket.append(path)
                if index > 80:
                    self.fail("could not form a view-disjoint train/val split")
            paths = train_paths[:2] + val_paths[:2]
            trajectories = root / "traj"
            summary = collect(
                paths,
                trajectories,
                episodes_per_state=8,
                experts=("rules", "greedy", "beam", "random"),
                levels=("clean", "light"),
                seed=17,
                val_ratio=val_ratio,
                max_steps=2,
                max_actions=64,
                image_size=32,
                shard_size=32,
                forbidden=set(),
                beam_width=2,
                beam_depth=1,
            )
            self.assertGreater(summary["transitions"], 0)
            self.assertEqual(summary["view_overlap"], [])
            self.assertGreaterEqual(summary["train_views"], 1)
            self.assertGreaterEqual(summary["val_views"], 1)
            self.assertEqual(
                set(summary["expert_transitions"]),
                {"rules", "greedy", "beam", "random"},
            )
            bc_dir = root / "bc"
            bc = train_bc(
                trajectories,
                bc_dir,
                experts=("rules", "greedy", "beam"),
                epochs=1,
                batch_size=4,
                hidden_dim=16,
                patience=1,
                seed=17,
                device_name="cpu",
            )
            self.assertTrue((bc_dir / "best.pt").is_file())
            self.assertGreater(bc["train_samples"], 0)
            gate = root / "bc_gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "checkpoint": str((bc_dir / "best.pt").resolve()),
                        "gate": {
                            "passed": True,
                            "rule": "smoke: PPO init from BC weights",
                        },
                    }
                ),
                encoding="utf-8",
            )
            ppo = train_ppo(
                paths,
                bc_dir / "best.pt",
                gate,
                root / "ppo",
                total_timesteps=8,
                num_envs=2,
                rollout_steps=2,
                update_epochs=1,
                minibatch_size=4,
                max_steps=2,
                max_actions=64,
                image_size=32,
                seed=17,
                device_name="cpu",
                subprocess=False,
            )
            self.assertEqual(ppo["kind"], "masked_ppo")
            self.assertGreater(ppo["global_step"], 0)
            self.assertTrue((root / "ppo" / "best.pt").is_file())
            self.assertTrue(isinstance(ppo["action_coverage"], dict))


if __name__ == "__main__":
    unittest.main()
