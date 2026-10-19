from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path

from layout_agent.evalset.human_gold import (
    T1_TARGET,
    T2_TARGET,
    load_tasks_js,
    read_jsonl,
    select_t1_gold,
    select_t2_gold,
    validate_labels,
)
from layout_agent.evalset.isolation import (
    colleague_challenge_views,
    forbidden_training_views,
    gold_views,
    human_pairs_test_views,
    human_pairs_train_views,
)


class HumanGoldTest(unittest.TestCase):
    def test_t2_prefers_high_pose_delta_from_human_pairs(self) -> None:
        pairs = [
            {
                "pair_id": "low",
                "view_key": "s1/a",
                "split": "test",
                "review_flag": "ok",
                "pose_delta": 2,
                "n_ann": 20,
                "quality": 9,
                "solution_id": "s1",
                "state_a": "a.json",
                "state_b": "b.json",
                "label": {"prefer": "b"},
            },
            {
                "pair_id": "high",
                "view_key": "s2/b",
                "split": "test",
                "review_flag": "ok",
                "pose_delta": 40,
                "n_ann": 8,
                "quality": 1,
                "solution_id": "s2",
                "state_a": "a.json",
                "state_b": "b.json",
                "label": {"prefer": "b"},
            },
        ]
        chosen = select_t2_gold(pairs, ["train"], target=1)
        self.assertEqual({row["id"] for row in chosen}, {"high"})
        self.assertEqual(chosen[0]["gold_source"], "human_pairs_test")

    def test_t2_rejects_train_leakage(self) -> None:
        with self.assertRaises(ValueError):
            select_t2_gold(
                [
                    {
                        "pair_id": "p1",
                        "view_key": "v1",
                        "split": "test",
                        "review_flag": "ok",
                        "pose_delta": 10,
                        "label": {"prefer": "b"},
                    }
                ],
                ["v1"],
                target=1,
            )

    def test_t1_is_one_per_view_and_excludes_colleague(self) -> None:
        tasks = [
            {"id": "h1", "view_key": "hold"},
            {"id": "h2", "view_key": "hold"},
            {"id": "v1", "view_key": "val"},
            {"id": "c1", "view_key": "col", "source": "colleague_pairs"},
        ]
        chosen = select_t1_gold(tasks, ["hold", "val", "col"], ["train"], target=2)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(len({row["view_key"] for row in chosen}), 2)
        self.assertNotIn("c1", {row["id"] for row in chosen})

    def test_t1_skips_discarded_and_stratifies_grades(self) -> None:
        tasks = [
            {"id": "d1", "view_key": "hold-d"},
            {"id": "g1", "view_key": "hold-1"},
            {"id": "g3", "view_key": "hold-3"},
            {"id": "g5", "view_key": "hold-5"},
        ]
        chosen = select_t1_gold(
            tasks,
            ["hold-d", "hold-1", "hold-3", "hold-5"],
            ["train"],
            target=3,
            discarded_ids=["d1"],
            grades={"g1": 1, "g3": 3, "g5": 5},
        )
        self.assertEqual({row["id"] for row in chosen}, {"g1", "g3", "g5"})
        self.assertNotIn("d1", {row["id"] for row in chosen})

    def test_t1_all_reviewed_can_promote_perturb_train_views(self) -> None:
        tasks = [
            {"id": "h1", "view_key": "hold"},
            {"id": "tr1", "view_key": "train-v"},
            {"id": "hp1", "view_key": "hp-test"},
        ]
        chosen = select_t1_gold(
            tasks,
            ["hold"],
            ["train-v"],
            target=None,
            grades={"h1": 4, "tr1": 2, "hp1": 5},
            blocked_view_keys=["hp-test"],
        )
        self.assertEqual({row["id"] for row in chosen}, {"h1", "tr1"})
        self.assertEqual(
            {row["id"]: row["gold_source"] for row in chosen},
            {"h1": "perturb_holdout", "tr1": "reviewed_perturb_train"},
        )

    def test_label_validation_filters_package_and_requires_adjudication(self) -> None:
        tasks = [
            {"id": "a", "gold_source": "perturb_val"},
            {"id": "b", "gold_source": "colleague_ood"},
        ]
        labels = [
            {
                "pair_id": "a",
                "label": {"prefer": "a", "annotator": "u1"},
            },
            {
                "pair_id": "a",
                "label": {"prefer": "a", "annotator": "u2"},
            },
            {
                "pair_id": "b",
                "label": {"prefer": "a", "annotator": "u1"},
            },
            {
                "pair_id": "b",
                "label": {"prefer": "b", "annotator": "u2"},
            },
            {
                "pair_id": "not-gold",
                "label": {"prefer": "a", "annotator": "u1"},
            },
        ]
        result = validate_labels("t2", tasks, labels)
        self.assertFalse(result.ready)
        self.assertEqual(result.conflicting_ids, ["b"])
        self.assertEqual(result.unknown_ids, ["not-gold"])
        self.assertEqual(result.discarded_ids, [])
        self.assertEqual(result.agreement["overlap_items"], 2)
        self.assertEqual(result.agreement["exact_agreements"], 1)

    def test_t1_grade_range_and_annotator_required(self) -> None:
        tasks = [{"id": "s", "gold_source": "holdout"}]
        with self.assertRaises(ValueError):
            validate_labels(
                "t1",
                tasks,
                [{"sample_id": "s", "label": {"grade": 6, "annotator": "u"}}],
            )
        with self.assertRaises(ValueError):
            validate_labels(
                "t1",
                tasks,
                [{"sample_id": "s", "label": {"grade": 3}}],
            )

    def test_exported_states_must_match_frozen_sides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left.json"
            right = root / "right.json"
            left.write_text("{}", encoding="utf-8")
            right.write_text("{}", encoding="utf-8")
            tasks = [
                {
                    "id": "p",
                    "view_key": "v",
                    "state_a": str(left),
                    "state_b": str(right),
                }
            ]
            with self.assertRaises(ValueError):
                validate_labels(
                    "t2",
                    tasks,
                    [
                        {
                            "pair_id": "p",
                            "view_key": "v",
                            "state_a": str(right),
                            "state_b": str(left),
                            "label": {"prefer": "a", "annotator": "u"},
                        }
                    ],
                )


ROOT = Path(__file__).resolve().parents[1]


class FrozenHumanProtocolTest(unittest.TestCase):
    def test_live_gold_is_80_t2_and_15_t1_with_no_leaks(self) -> None:
        gold = ROOT / "data" / "evalsys" / "human_gold"
        manifest = json.loads((gold / "manifest.json").read_text(encoding="utf-8"))
        public = json.loads(
            (ROOT / "data" / "evalsys" / "test_manifest.json").read_text(encoding="utf-8")
        )
        t1 = read_jsonl(gold / "t1_tasks.jsonl")
        t2 = read_jsonl(gold / "t2_tasks.jsonl")
        self.assertGreaterEqual(len(t1), T1_TARGET)
        self.assertEqual(len(t2), T2_TARGET)
        self.assertGreaterEqual(public["n_t1_tasks"], T1_TARGET)
        self.assertEqual(public["n_t2_pairs"], T2_TARGET)
        self.assertGreaterEqual(len({row["view_key"] for row in t1}), T1_TARGET)
        t1_views = {str(row.get("view_key") or "") for row in t1}
        t2_views = {str(row.get("view_key") or "") for row in t2}
        self.assertTrue(set(manifest.get("train_view_overlap") or []) <= t1_views)
        self.assertFalse(t2_views & set(manifest.get("train_view_overlap") or []))
        self.assertEqual(manifest["status"], "human_labels_frozen")
        self.assertTrue(manifest["labels"]["t1"])
        self.assertTrue(manifest["labels"]["t2"])
        colleague = colleague_challenge_views(ROOT)
        selected = gold_views(gold)
        self.assertTrue(colleague)
        self.assertFalse(selected & colleague)
        forbidden = forbidden_training_views(ROOT)
        self.assertTrue(selected <= forbidden)
        pair_test = human_pairs_test_views(ROOT)
        pair_train = human_pairs_train_views(ROOT)
        self.assertTrue(pair_test)
        self.assertTrue(pair_test <= forbidden)
        t1_views = {str(row.get("view_key") or "") for row in t1}
        t2_views = {str(row.get("view_key") or "") for row in t2}
        self.assertTrue(t2_views <= pair_test)
        self.assertFalse(t1_views & pair_test)
        self.assertFalse(selected & pair_train)
        self.assertTrue(all(row.get("gold_source") == "human_pairs_test" for row in t2))
        t1_labels = read_jsonl(gold / "t1_labels.jsonl")
        t2_labels = read_jsonl(gold / "t2_labels.jsonl")
        self.assertEqual(len(t1_labels), len(t1))
        self.assertGreaterEqual(len(t1_labels), T1_TARGET)
        self.assertEqual(len(t2_labels), T2_TARGET)
        grades = {int(row["label"]["grade"]) for row in t1_labels}
        self.assertGreaterEqual(len(grades), 3)
        self.assertEqual(
            load_tasks_js(ROOT / "data" / "evalsys" / "annotate_t2_gold_supplement" / "tasks.js"),
            [],
        )

    def test_validate_without_label_files_is_not_ready(self) -> None:
        from layout_agent.evalset.human_gold import LabelValidation

        result = validate_labels("t2", [{"id": "p1", "gold_source": "perturb_val"}], [])
        self.assertIsInstance(result, LabelValidation)
        self.assertFalse(result.ready)
        self.assertEqual(result.missing_ids, ["p1"])
        self.assertEqual(result.discarded_ids, [])

    def test_discard_is_not_a_gold_preference(self) -> None:
        result = validate_labels(
            "t2",
            [{"id": "a", "gold_source": "perturb_val"}],
            [
                {
                    "pair_id": "a",
                    "label": {
                        "discard": True,
                        "discard_reason": "weird",
                        "annotator": "u",
                    },
                }
            ],
        )
        self.assertFalse(result.ready)
        self.assertEqual(result.discarded_ids, ["a"])
        self.assertEqual(result.missing_ids, ["a"])
        self.assertEqual(result.frozen, [])


if __name__ == "__main__":
    unittest.main()
