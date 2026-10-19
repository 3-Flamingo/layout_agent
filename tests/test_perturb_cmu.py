from __future__ import annotations

import random
import unittest

from layout_agent.evalset.layout_diff import pose_delta
from layout_agent.evalset.perturb import (
    measurement_signature,
    perturb_state_cmu_complex,
    placement_outside_frame,
)
from layout_agent.models import LayoutState, LinearAnno, Point, RadialAnno, RasterMeta


def _state() -> LayoutState:
    return LayoutState(
        state_id="perfect",
        raster=RasterMeta(
            instance_image="missing.png",
            width=8,
            height=8,
            origin=Point(x=0, y=80),
            pixel_size=1.0,
        ),
        color_legend={"entries": {}},
        view={"bbox": {"min": {"x": 0, "y": 0}, "max": {"x": 80, "y": 60}}},
        linears=[
            LinearAnno(
                id="H1",
                color="ff0000",
                p1=Point(x=10, y=20),
                p2=Point(x=50, y=20),
                dir=Point(x=0, y=1),
                point=Point(x=30, y=28),
                value=40,
                text_point=Point(x=30, y=30),
            ),
            LinearAnno(
                id="H2",
                color="ff0000",
                p1=Point(x=10, y=40),
                p2=Point(x=50, y=40),
                dir=Point(x=0, y=1),
                point=Point(x=30, y=48),
                value=40,
                text_point=Point(x=30, y=50),
            ),
            LinearAnno(
                id="V1",
                color="00ff00",
                p1=Point(x=10, y=10),
                p2=Point(x=10, y=50),
                dir=Point(x=-1, y=0),
                point=Point(x=2, y=30),
                value=40,
                text_point=Point(x=0, y=30),
            ),
            LinearAnno(
                id="V2",
                color="00ff00",
                p1=Point(x=50, y=10),
                p2=Point(x=50, y=50),
                dir=Point(x=1, y=0),
                point=Point(x=58, y=30),
                value=40,
                text_point=Point(x=60, y=30),
            ),
        ],
        diameters=[
            RadialAnno(
                id="D1",
                category="diameter",
                color="0000ff",
                anchor=Point(x=30, y=30),
                path=[Point(x=30, y=30), Point(x=42, y=36)],
            )
        ],
    )


class CmuComplexPerturbTest(unittest.TestCase):
    def test_locks_measurement_objects(self) -> None:
        src = _state()
        before = measurement_signature(src)
        out, ops = perturb_state_cmu_complex(src, random.Random(7))
        self.assertTrue(ops)
        self.assertEqual(measurement_signature(out), before)
        self.assertEqual(measurement_signature(src), before)
        self.assertGreaterEqual(pose_delta(src, out), 2)
        self.assertEqual(src.linears[0].point.x, 30.0)

    def test_seed_is_deterministic(self) -> None:
        src = _state()
        a, ops_a = perturb_state_cmu_complex(src, random.Random(11))
        b, ops_b = perturb_state_cmu_complex(src, random.Random(11))
        self.assertEqual(ops_a, ops_b)
        self.assertEqual(a.linears[0].point, b.linears[0].point)
        self.assertEqual(a.diameters[0].path[-1], b.diameters[0].path[-1])

    def test_stays_inside_view_frame(self) -> None:
        src = _state()
        out, ops = perturb_state_cmu_complex(src, random.Random(7))
        self.assertTrue(ops)
        self.assertEqual(placement_outside_frame(out), [])
        self.assertGreaterEqual(pose_delta(src, out), 2)


if __name__ == "__main__":
    unittest.main()
