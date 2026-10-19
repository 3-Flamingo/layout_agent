"""Unit tests for CAD tolerance strings and roughness glyphs."""

from __future__ import annotations

import unittest

from layout_agent.cad_marks import (
    cad_to_fixed,
    format_cad_tolerance,
    roughness_local_geom,
    roughness_world_geom,
)
from layout_agent.models import Point, RoughnessAnno


class CadMarksTest(unittest.TestCase):
    def test_cad_to_fixed_strips_zeros(self) -> None:
        self.assertEqual(cad_to_fixed(0.05, 2), "0.05")
        self.assertEqual(cad_to_fixed(0.10, 2), "0.1")
        self.assertEqual(cad_to_fixed(0.0, 2), "0")

    def test_symmetric_tolerance(self) -> None:
        tol = format_cad_tolerance(
            {"dimtol": True, "dimtm": 0.05, "dimtp": 0.05, "dimtfac": 0.5, "dimtolj": 0, "precision": 2},
            5.0,
        )
        self.assertIsNotNone(tol)
        assert tol is not None
        self.assertEqual(tol["kind"], "symmetric")
        self.assertEqual(tol["down"], "±0.05")
        self.assertEqual(tol["up"], "")
        self.assertAlmostEqual(tol["height"], 2.5)

    def test_asymmetric_zero_minus(self) -> None:
        tol = format_cad_tolerance(
            {"dimtol": True, "dimtm": 0.1, "dimtp": 0, "dimtfac": 0.5, "precision": 2},
            5.0,
        )
        self.assertIsNotNone(tol)
        assert tol is not None
        self.assertEqual(tol["kind"], "deviation")
        self.assertEqual(tol["up"], "0")
        self.assertEqual(tol["down"], "-0.1")

    def test_no_tolerance_when_dimtol_off(self) -> None:
        self.assertIsNone(format_cad_tolerance({"precision": 2}, 5.0))
        self.assertIsNone(format_cad_tolerance({"dimtol": True, "precision": 2}, 5.0))

    def test_style1_check_and_value(self) -> None:
        geom = roughness_local_geom(1, "3.2", 2.5, rotation=0)
        self.assertEqual(len(geom.polylines), 1)
        self.assertEqual(len(geom.polylines[0]), 4)
        self.assertEqual(geom.texts[0][2], "3.2")
        self.assertEqual(geom.texts[0][3], 0.0)
        flipped = roughness_local_geom(1, "3.2", 2.5, rotation=180)
        self.assertEqual(flipped.texts[0][3], 180.0)

    def test_world_insert_keeps_tip_at_insert(self) -> None:
        anno = RoughnessAnno(id="r", insert=Point(x=10, y=20), rotation=0, value="3.2", style=1)
        world = roughness_world_geom(anno)
        tips = [(x, y) for poly in world.polylines for x, y in poly]
        self.assertTrue(any(abs(x - 10) < 1e-6 and abs(y - 20) < 1e-6 for x, y in tips))


if __name__ == "__main__":
    unittest.main()
