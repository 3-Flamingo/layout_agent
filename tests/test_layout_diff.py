from __future__ import annotations

import unittest

from layout_agent.evalset.layout_diff import (
    annotation_count,
    layouts_identical,
    pose_delta,
)
from layout_agent.models import LayoutState, LinearAnno, Point, RadialAnno, RasterMeta


def _state(linears, diameters=None) -> LayoutState:
    return LayoutState(
        state_id="x",
        raster=RasterMeta(
            instance_image="missing.png",
            width=8,
            height=8,
            origin=Point(x=0, y=8),
            pixel_size=1.0,
        ),
        color_legend={"entries": {}},
        linears=linears,
        diameters=diameters or [],
    )


class LayoutDiffTest(unittest.TestCase):
    def test_identical_ignores_id_and_color(self) -> None:
        a = _state(
            [
                LinearAnno(
                    id="L1",
                    color="ff0000",
                    p1=Point(x=0, y=0),
                    p2=Point(x=10, y=0),
                    dir=Point(x=0, y=1),
                    point=Point(x=5, y=4),
                    value=10,
                )
            ]
        )
        b = _state(
            [
                LinearAnno(
                    id="OTHER",
                    color="00ff00",
                    p1=Point(x=0, y=0),
                    p2=Point(x=10, y=0),
                    dir=Point(x=0, y=1),
                    point=Point(x=5, y=4),
                    value=10,
                )
            ]
        )
        self.assertTrue(layouts_identical(a, b))
        self.assertEqual(pose_delta(a, b), 0)

    def test_moved_dim_is_different(self) -> None:
        a = _state(
            [
                LinearAnno(
                    id="L1",
                    color="ff0000",
                    p1=Point(x=0, y=0),
                    p2=Point(x=10, y=0),
                    dir=Point(x=0, y=1),
                    point=Point(x=5, y=4),
                    value=10,
                )
            ]
        )
        b = _state(
            [
                LinearAnno(
                    id="L1",
                    color="ff0000",
                    p1=Point(x=0, y=0),
                    p2=Point(x=10, y=0),
                    dir=Point(x=0, y=-1),
                    point=Point(x=5, y=-6),
                    value=10,
                )
            ]
        )
        self.assertFalse(layouts_identical(a, b))
        self.assertGreater(pose_delta(a, b), 0)
        self.assertEqual(annotation_count(a), 1)

    def test_radial_path_relative_to_anchor(self) -> None:
        a = _state(
            [],
            [
                RadialAnno(
                    id="D1",
                    category="diameter",
                    color="0000ff",
                    anchor=Point(x=2, y=2),
                    path=[Point(x=2, y=2), Point(x=8, y=2)],
                )
            ],
        )
        b = _state(
            [],
            [
                RadialAnno(
                    id="D9",
                    category="diameter",
                    color="ffffff",
                    anchor=Point(x=2, y=2),
                    path=[Point(x=2, y=2), Point(x=8, y=2)],
                )
            ],
        )
        self.assertTrue(layouts_identical(a, b))


if __name__ == "__main__":
    unittest.main()
