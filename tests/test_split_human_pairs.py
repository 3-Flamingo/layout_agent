from __future__ import annotations

import unittest

from layout_agent.evalset.split_human_pairs import assign_splits, random_test_keys


class SplitHumanPairsTest(unittest.TestCase):
    def test_random_split_is_seeded_and_sized(self) -> None:
        keys = [f"sol/{i}" for i in range(20)]
        a = random_test_keys(keys, n_test=5, seed=7)
        b = random_test_keys(keys, n_test=5, seed=7)
        c = random_test_keys(keys, n_test=5, seed=8)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 5)
        self.assertEqual(len(set(a)), 5)
        self.assertNotEqual(a, c)

    def test_n_test_is_fraction_of_all_but_capped_by_real(self) -> None:
        from layout_agent.evalset.split_human_pairs import n_test_from_all

        self.assertEqual(n_test_from_all(425, 425, ratio=0.20), 170)
        self.assertEqual(n_test_from_all(10, 0, ratio=0.20), 2)
        self.assertEqual(n_test_from_all(3, 100, ratio=0.20), 3)

    def test_assign_splits_relabels_rows(self) -> None:
        rows = [
            {"pair_id": "p2", "view_key": "sol/b", "split": "train"},
            {"pair_id": "p1", "view_key": "sol/a", "split": "test"},
        ]
        test_rows, train_rows = assign_splits(rows, ["sol/a"])
        self.assertEqual([r["pair_id"] for r in test_rows], ["p1"])
        self.assertEqual([r["pair_id"] for r in train_rows], ["p2"])
        self.assertEqual(test_rows[0]["split"], "test")
        self.assertEqual(train_rows[0]["split"], "train")


if __name__ == "__main__":
    unittest.main()
