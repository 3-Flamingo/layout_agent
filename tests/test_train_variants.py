from __future__ import annotations

import unittest

from layout_agent.evalset.train_variants import (
    build_train_with_synth,
    prepare_synth_train_row,
)


class TrainVariantsTest(unittest.TestCase):
    def test_synth_row_is_train_labeled(self) -> None:
        row = prepare_synth_train_row(
            {
                "pair_id": "cmu_x",
                "view_key": "sol/v",
                "split": "cmu_perturb",
                "review_only": True,
                "label": {"prefer": "b"},
            }
        )
        self.assertEqual(row["split"], "train")
        self.assertTrue(row["synthetic"])
        self.assertEqual(row["synth_kind"], "cmu_perturb")
        self.assertNotIn("review_only", row)

    def test_merge_keeps_real_and_drops_blocked(self) -> None:
        real = [
            {"pair_id": "r1", "view_key": "sol/a", "split": "train"},
            {"pair_id": "r2", "view_key": "sol/b", "split": "train"},
        ]
        synth = [
            {"pair_id": "cmu_a", "view_key": "sol/a", "split": "cmu_perturb", "review_only": True},
            {"pair_id": "cmu_test", "view_key": "sol/test", "split": "cmu_perturb"},
            {"pair_id": "cmu_other", "view_key": "sol/z", "split": "cmu_perturb"},
        ]
        merged, stats = build_train_with_synth(
            real, synth, blocked_view_keys={"sol/test"}
        )
        self.assertEqual(stats["real"], 2)
        self.assertEqual(stats["synth_kept"], 1)
        self.assertEqual(stats["synth_blocked"], 1)
        self.assertEqual(stats["synth_outside_train"], 1)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["pair_id"], "r1")
        self.assertEqual(merged[2]["pair_id"], "cmu_a")
        self.assertTrue(merged[2]["synthetic"])
        self.assertEqual(merged[0].get("synthetic"), None)


if __name__ == "__main__":
    unittest.main()
