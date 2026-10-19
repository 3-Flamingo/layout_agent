from __future__ import annotations

import unittest

from layout_agent.training.metrics import pairwise_agreement


class PairwiseAgreementTest(unittest.TestCase):
    def test_reports_strict_accuracy_and_coverage(self) -> None:
        result = pairwise_agreement(
            [
                (2.0, 1.0, "a"),
                (1.0, 2.0, "a"),
                (1.0, 1.0, "a"),
                (1.0, 2.0, "tie"),
            ]
        )

        self.assertEqual(result["n"], 4)
        self.assertEqual(result["agree"], 1)
        self.assertEqual(result["disagree"], 1)
        self.assertEqual(result["model_tie"], 1)
        self.assertEqual(result["human_tie"], 1)
        self.assertEqual(result["agreement_rate"], 0.5)
        self.assertEqual(result["strict_accuracy"], 0.25)
        self.assertEqual(result["decided_coverage"], 0.5)

    def test_margin_controls_model_ties(self) -> None:
        result = pairwise_agreement([(1.01, 1.0, "a")], margin=0.02)

        self.assertEqual(result["model_tie"], 1)
        self.assertEqual(result["strict_accuracy"], 0.0)
        self.assertEqual(result["decided_coverage"], 0.0)

    def test_force_choice_breaks_near_ties(self) -> None:
        result = pairwise_agreement(
            [(1.01, 1.0, "a")],
            margin=0.02,
            force_choice=True,
        )
        self.assertEqual(result["model_tie"], 0)
        self.assertEqual(result["agree"], 1)
        self.assertEqual(result["strict_accuracy"], 1.0)

    def test_ranking_key_compares_lexicographically(self) -> None:
        result = pairwise_agreement(
            [
                ((90.001, -1.0), (90.0, -0.5), "a"),
                ((80.0, -3.0), (80.0, -3.0), "b"),
            ],
            force_choice=True,
        )
        self.assertEqual(result["agree"], 1)
        self.assertEqual(result["model_tie"], 1)
        self.assertEqual(result["strict_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
