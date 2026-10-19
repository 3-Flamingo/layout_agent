from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from layout_agent.models import ActionType
from layout_agent.training.env import LayoutEnv


def _state_dict() -> dict:
    return {
        "state_id": "rl_fixture",
        "source": {"solution_id": "fixture", "view_uuid": "v1"},
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
                "ff00ff": {"id": "lin3", "category": "linear"},
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
            {
                "id": "lin3",
                "color": "ff00ff",
                "p1": {"x": 25.0, "y": 60.0},
                "p2": {"x": 75.0, "y": 60.0},
                "dir": {"x": 0.0, "y": 1.0},
                "point": {"x": 50.0, "y": 68.0},
                "value": 50.0,
                "text": {"content": "50", "height": 2.5},
            }
        ],
        "diameters": [
            {
                "id": "dia",
                "category": "diameter",
                "color": "00ff00",
                "anchor": {"x": 50.0, "y": 50.0},
                "path": [
                    {"x": 50.0, "y": 50.0},
                    {"x": 72.0, "y": 50.0},
                ],
                "text": {"content": "⌀10", "height": 2.5},
            }
        ],
    }


class LayoutEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state_path = self.root / "fixture.state.json"
        self.state_path.write_text(json.dumps(_state_dict()), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _env(self, name: str) -> LayoutEnv:
        return LayoutEnv(
            [self.state_path],
            img_size=64,
            max_actions=64,
            max_steps=4,
            work_dir=self.root / name,
            seed=11,
        )

    def test_fixed_schema_and_all_current_action_families(self) -> None:
        env = self._env("schema")
        observation, info = env.reset(seed=11, options={"index": 0})
        self.assertTrue(env.observation_space.contains(observation))
        kinds = {action.type for action in env.legal_actions}
        self.assertTrue(
            {
                ActionType.ACCEPT,
                ActionType.BALANCE_LINEAR_SIDES,
                ActionType.SET_LINEAR_LAYER_ORDER,
                ActionType.SET_LINEAR_SIDE,
                ActionType.SET_RADIAL_SECTOR,
            }.issubset(kinds)
        )
        self.assertEqual(info["legal_action_count"], int(observation["action_mask"].sum()))

    def test_current_side_and_sector_are_masked(self) -> None:
        env = self._env("mask")
        env.reset(seed=11, options={"index": 0})
        linear_sides = {
            action.side
            for action in env.legal_actions
            if action.type == ActionType.SET_LINEAR_SIDE
        }
        radial_sectors = {
            action.sector
            for action in env.legal_actions
            if action.type == ActionType.SET_RADIAL_SECTOR
        }
        self.assertNotIn("U", linear_sides)
        self.assertIn("D", linear_sides)
        self.assertNotIn(0, radial_sectors)
        self.assertEqual(len(radial_sectors), 7)
        for action in env.legal_actions:
            if action.type != ActionType.SET_RADIAL_SECTOR:
                continue
            candidate, _ = env.executor.apply_geometry(env.state, [action])
            radial = candidate.diameters[0]
            end = radial.path[-1]
            angle = math.atan2(
                end.y - radial.anchor.y,
                end.x - radial.anchor.x,
            ) % (2.0 * math.pi)
            actual = int(round(angle / (math.pi / 4.0))) % 8
            self.assertEqual(actual, action.sector)

    def test_replay_is_deterministic_and_legal_frame_is_immutable(self) -> None:
        env_a = self._env("replay_a")
        env_b = self._env("replay_b")
        _, reset_a = env_a.reset(seed=23, options={"index": 0})
        _, reset_b = env_b.reset(seed=23, options={"index": 0})
        self.assertEqual(reset_a["state_hash"], reset_b["state_hash"])
        key = next(
            i
            for i, action in enumerate(env_a.legal_actions)
            if action.type == ActionType.SET_LINEAR_SIDE
        )
        key_b = next(
            i
            for i, action in enumerate(env_b.legal_actions)
            if action.model_dump(mode="json")
            == env_a.legal_actions[key].model_dump(mode="json")
        )
        original_bbox = json.loads(json.dumps(env_a.state.view["bbox"]))
        _, reward_a, term_a, trunc_a, info_a = env_a.step(key)
        _, reward_b, term_b, trunc_b, info_b = env_b.step(key_b)
        self.assertEqual(reward_a, reward_b)
        self.assertEqual((term_a, trunc_a), (term_b, trunc_b))
        self.assertEqual(info_a["state_hash"], info_b["state_hash"])
        self.assertEqual(info_a["reward_terms"], info_b["reward_terms"])
        self.assertEqual(env_a.state.view["bbox"], original_bbox)
        self.assertEqual(env_a.state.view["legal_bbox"], original_bbox)

    def test_masked_action_index_is_rejected(self) -> None:
        env = self._env("invalid")
        observation, _ = env.reset(options={"index": 0})
        invalid = int(observation["action_mask"].sum())
        with self.assertRaises(ValueError):
            env.step(invalid)

    def test_render_scale_stays_fixed_when_action_remains_inside_frame(self) -> None:
        env = self._env("render_scale")
        env.reset(options={"index": 0})
        before = (env.state.raster.width, env.state.raster.height)
        index = next(
            i
            for i, action in enumerate(env.legal_actions)
            if action.type == ActionType.SET_RADIAL_SECTOR and action.sector == 1
        )
        candidate, _ = env.executor.apply(
            env.state,
            [env.legal_actions[index]],
            self.root / "render_scale_direct",
        )
        self.assertNotIn("render_bbox", candidate.view)
        self.assertEqual((candidate.raster.width, candidate.raster.height), before)

    def test_forbidden_view_cannot_enter_training(self) -> None:
        env = LayoutEnv(
            [self.state_path],
            img_size=32,
            max_actions=64,
            work_dir=self.root / "forbidden",
            forbidden_view_keys=["fixture/v1"],
        )
        with self.assertRaises(ValueError):
            env.reset(options={"index": 0})

    def test_h_oob_uses_legal_frame_not_render_bbox(self) -> None:
        env = self._env("oob")
        env.reset(options={"index": 0})
        before = sum(
            1 for item in env.report.hard_violations if item.code == "H_OOB"
        )
        env.state.view["render_bbox"] = {
            "min": {"x": 49.0, "y": 49.0},
            "max": {"x": 51.0, "y": 51.0},
        }
        env.state.view["bbox"] = {
            "min": {"x": -1000.0, "y": -1000.0},
            "max": {"x": 1000.0, "y": 1000.0},
        }
        after = env.evaluator.evaluate(env.state)
        self.assertEqual(
            before,
            sum(1 for item in after.hard_violations if item.code == "H_OOB"),
        )

    def test_every_current_action_family_can_execute(self) -> None:
        env = self._env("families")
        wanted = {
            ActionType.ACCEPT,
            ActionType.BALANCE_LINEAR_SIDES,
            ActionType.SET_LINEAR_LAYER_ORDER,
            ActionType.SET_LINEAR_SIDE,
            ActionType.SET_RADIAL_SECTOR,
        }
        executed: set[ActionType] = set()
        for action_type in wanted:
            env.reset(seed=11, options={"index": 0})
            index = next(
                i
                for i, action in enumerate(env.legal_actions)
                if action.type == action_type
            )
            _, _, terminated, truncated, info = env.step(index)
            executed.add(ActionType(info["action"]["type"]))
            self.assertTrue(terminated or truncated or env.state.view["legal_bbox"])
        self.assertEqual(executed, wanted)

    def test_ungated_human_reward_is_refused(self) -> None:
        gate = self.root / "gate.json"
        gate.write_text(json.dumps({"passed": False}), encoding="utf-8")
        checkpoint = self.root / "reward.pt"
        checkpoint.write_bytes(b"x")
        with self.assertRaises(ValueError):
            LayoutEnv(
                [self.state_path],
                img_size=32,
                max_actions=64,
                work_dir=self.root / "human",
                human_reward_checkpoint=checkpoint,
                human_reward_gate=gate,
                human_reward_beta=0.01,
            )


if __name__ == "__main__":
    unittest.main()
