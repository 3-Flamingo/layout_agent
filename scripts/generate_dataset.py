#!/usr/bin/env python3
"""Generate Layout Agent dataset (ICM + ColorLegend + LayoutState).

Synthetic geometry is styled after 2d_service dimension layout objects
(aligned / diameter / radial + projection obstacles), without requiring
the TypeScript CAD runtime.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.config import load_config
from layout_agent.models import (
    ColorLegend,
    ColorLegendEntry,
    LayoutState,
    LinearAnno,
    Obstacle,
    Point,
    RadialAnno,
    RasterMeta,
)
from layout_agent.perceive import ACTIVE_PALETTE, Perceiver


def build_case(case_id: str, kind: str = "conflict") -> LayoutState:
    """kind: clean | conflict | crowded"""
    bbox_min = Point(x=0, y=0)
    bbox_max = Point(x=200, y=120)
    obstacles = [
        Obstacle(
            id="proj_loop",
            kind="polyline",
            raw_type="projection",
            polyline=[
                Point(x=30, y=25),
                Point(x=170, y=25),
                Point(x=170, y=95),
                Point(x=30, y=95),
                Point(x=30, y=25),
            ],
            fill_role="loop",
        )
    ]

    c0, c1, c2, c3 = ACTIVE_PALETTE[:4]
    linears = [
        LinearAnno(
            id="L_overall",
            color=c0,
            p1=Point(x=30, y=25),
            p2=Point(x=170, y=25),
            dir=Point(x=0, y=-1),
            point=Point(x=100, y=16),
            value=140,
            text={"content": "140", "width": 12, "height": 3.5},
        ),
        LinearAnno(
            id="L_seg",
            color=c1,
            p1=Point(x=30, y=25),
            p2=Point(x=100, y=25),
            dir=Point(x=0, y=1),
            point=Point(x=65, y=105 if kind != "crowded" else 12),
            value=70,
            text={"content": "70", "width": 10, "height": 3.5},
        ),
    ]

    diameters = [
        RadialAnno(
            id="D_hole",
            category="diameter",
            color=c2,
            anchor=Point(x=80, y=60),
            path=[
                Point(x=80, y=60),
                Point(x=80, y=108),
                Point(x=115, y=108),
            ],
            text={"content": "⌀10", "width": 12, "height": 5},
        )
    ]
    radials = [
        RadialAnno(
            id="R_fillet",
            category="radial",
            color=c3,
            anchor=Point(x=150, y=40),
            path=[
                Point(x=150, y=40),
                Point(x=188, y=40),
                Point(x=188, y=15),
            ],
            text={"content": "R5", "width": 10, "height": 5},
        )
    ]

    if kind == "clean":
        # separate sides: overall below, segment left, diameter right-up, radial right
        linears[1].point = Point(x=18, y=60)
        linears[1].dir = Point(x=-1, y=0)
        linears[1].p1 = Point(x=30, y=40)
        linears[1].p2 = Point(x=30, y=80)
        diameters[0].path = [
            Point(x=80, y=60),
            Point(x=120, y=60),
            Point(x=145, y=75),
        ]

    if kind == "conflict":
        # diameter text deliberately overlaps overall linear text (outside part)
        diameters[0].path = [
            Point(x=80, y=60),
            Point(x=100, y=8),
            Point(x=100, y=8),
        ]
        diameters[0].text = {"content": "⌀10", "width": 22, "height": 8}
        linears[0].point = Point(x=100, y=10)
        linears[0].text = {"content": "140", "width": 22, "height": 8}
        # keep L_seg on left to isolate diameter vs overall clash
        linears[1].point = Point(x=18, y=60)
        linears[1].dir = Point(x=-1, y=0)
        linears[1].p1 = Point(x=30, y=40)
        linears[1].p2 = Point(x=30, y=80)
    if kind == "crowded":
        linears.append(
            LinearAnno(
                id="L_mid",
                color=ACTIVE_PALETTE[4],
                p1=Point(x=50, y=25),
                p2=Point(x=140, y=25),
                dir=Point(x=0, y=-1),
                point=Point(x=95, y=15),
                value=90,
                text={"content": "90", "width": 10, "height": 3.5},
            )
        )

    legend = ColorLegend(entries={})
    for a in linears:
        legend.entries[a.color] = ColorLegendEntry(id=a.id, category="linear")
    for a in diameters:
        legend.entries[a.color] = ColorLegendEntry(id=a.id, category="diameter")
    for a in radials:
        legend.entries[a.color] = ColorLegendEntry(id=a.id, category="radial")

    state = LayoutState(
        state_id=case_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        source={
            "drawing_id": f"SYN_{case_id}",
            "view_id": "V_front",
            "view_uuid": f"uuid-{case_id}",
            "view_type": "Front",
            "origin": "synthetic_from_2d_service_semantics",
        },
        meta={
            "text_height": 3.5,
            "revision": 0,
            "phase_tag": "base",
            "twod_ref": "DefaultDimensionsLayoutStrategyPlugin / DLAligned+DLRadial",
        },
        view={"bbox": {"min": bbox_min.model_dump(), "max": bbox_max.model_dump()}},
        raster=RasterMeta(
            instance_image="placeholder.png",
            width=1,
            height=1,
            origin=Point(x=0, y=0),
            pixel_size=0.25,
        ),
        color_legend=legend,
        obstacles=obstacles,
        linears=linears,
        diameters=diameters,
        radials=radials,
        linear_groups=[
            {
                "group_id": "default_linear",
                "dir": {"x": 0, "y": -1},
                "member_ids": [x.id for x in linears],
            }
        ],
    )
    return state


def main():
    cfg = load_config()
    out_root = ROOT / cfg.get("paths", {}).get("dataset_root", "data/datasets/demo")
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    snap = out_root / "snapshots"
    snap.mkdir(parents=True, exist_ok=True)

    perceiver = Perceiver(cfg)
    cases = [
        ("case_clean", "clean"),
        ("case_conflict", "conflict"),
        ("case_crowded", "crowded"),
    ]
    manifest_items = []

    for case_id, kind in cases:
        state = build_case(case_id, kind)
        rel_inst = f"snapshots/{case_id}.instance.png"
        rel_obs = f"snapshots/{case_id}.obstacle.png"
        rel_state = f"snapshots/{case_id}.state.json"
        rel_legend = f"snapshots/{case_id}.legend.json"

        abs_inst = out_root / rel_inst
        abs_obs = out_root / rel_obs
        state = perceiver.render_icm(state, abs_inst, abs_obs)

        # store relative paths in state for portability
        state.raster.instance_image = rel_inst
        state.raster.obstacle_image = rel_obs

        legend_path = out_root / rel_legend
        with open(legend_path, "w", encoding="utf-8") as f:
            json.dump(state.color_legend.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        # embed legend as well
        state_path = out_root / rel_state
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        tags = ["synthetic", kind]
        if state.diameters:
            tags.append("has_diameter")
        if state.linears:
            tags.append("has_linear")
        manifest_items.append(
            {
                "state_path": rel_state,
                "instance_image_path": rel_inst,
                "legend_path": rel_legend,
                "obstacle_image_path": rel_obs,
                "tags": tags,
                "split": "train" if kind != "clean" else "holdout",
            }
        )
        print(f"generated {case_id} -> {state_path}")

    manifest = {
        "dataset_name": "agent1_demo_synthetic",
        "schema_version": "agent1.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "owner": "layout_agent",
        "note": "Synthetic data aligned with 2d_service dim layout semantics (linear/diameter/radial + projection obstacles).",
        "twod_service_root": cfg.get("paths", {}).get("twod_service_root"),
        "items": manifest_items,
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"manifest -> {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
