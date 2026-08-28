"""Perception / State Builder: load or render ICM + ColorLegend + geometry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageChops, ImageDraw

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


# Distinct palette for Active annotations (no anti-alias intermediates)
ACTIVE_PALETTE = [
    "e6194b", "3cb44b", "4363d8", "f58231", "911eb4",
    "42d4f4", "f032e6", "bfef45", "fabed4", "469990",
    "dcbeff", "9a6324", "fffac8", "800000", "aaffc3",
]


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lower().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def world_to_pixel(
    x: float,
    y: float,
    origin: Point,
    pixel_size: float,
    height: int,
    y_up: bool = True,
) -> Tuple[int, int]:
    col = int((x - origin.x) / pixel_size)
    if y_up:
        row = int((origin.y - y) / pixel_size)
    else:
        row = int((y - origin.y) / pixel_size)
    return col, row


class Perceiver:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        p = self.cfg.get("perceive", {})
        self.pixel_size = float(p.get("pixel_size", 0.25))
        self.background = p.get("background", "000000")
        self.obstacle_reserved = list(p.get("obstacle_reserved", ["ffffff"]))

    def load_state(self, state_path: Path, dataset_root: Path) -> LayoutState:
        with open(state_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # resolve relative image paths against dataset root
        raster = raw.get("raster") or {}
        for key in ("instance_image", "obstacle_image"):
            p = raster.get(key)
            if p and not Path(p).is_absolute():
                raster[key] = str((dataset_root / p).as_posix())
        raw["raster"] = raster
        return LayoutState.model_validate(raw)

    def render_icm(
        self,
        state: LayoutState,
        out_instance: Path,
        out_obstacle: Optional[Path] = None,
    ) -> LayoutState:
        """Re-render ICM from geometry (used after execute). Keeps color↔id binding."""
        bbox = state.view.get("bbox") or {}
        bmin = bbox.get("min", {"x": 0, "y": 0})
        bmax = bbox.get("max", {"x": 100, "y": 80})
        pad = 5.0
        xmin, ymin = float(bmin["x"]) - pad, float(bmin["y"]) - pad
        xmax, ymax = float(bmax["x"]) + pad, float(bmax["y"]) + pad

        ps = self.pixel_size
        width = max(8, int((xmax - xmin) / ps) + 1)
        height = max(8, int((ymax - ymin) / ps) + 1)
        origin = Point(x=xmin, y=ymax)  # top-left in y_up world

        bg = hex_to_rgb(self.background)
        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)
        obs_img = Image.new("L", (width, height), 0)
        obs_draw = ImageDraw.Draw(obs_img)

        # Per-id masks so overwrite on composite still yields pixel-level conflicts
        id_masks: Dict[str, Image.Image] = {}
        id_draws: Dict[str, Any] = {}

        def ensure_mask(aid: str):
            if aid not in id_masks:
                id_masks[aid] = Image.new("L", (width, height), 0)
                id_draws[aid] = ImageDraw.Draw(id_masks[aid])

        def line(pts: List[Point], color: str, aid: str, width_px: int = 2):
            ensure_mask(aid)
            rgb = hex_to_rgb(color)
            pix = [world_to_pixel(p.x, p.y, origin, ps, height) for p in pts]
            if len(pix) >= 2:
                draw.line(pix, fill=rgb, width=width_px)
                id_draws[aid].line(pix, fill=255, width=width_px)
            for c, r in pix:
                draw.ellipse([c - 1, r - 1, c + 1, r + 1], fill=rgb)
                id_draws[aid].ellipse([c - 1, r - 1, c + 1, r + 1], fill=255)

        def text_blob(center: Point, color: str, aid: str, tw: float = 8, th: float = 3):
            ensure_mask(aid)
            rgb = hex_to_rgb(color)
            c0, r0 = world_to_pixel(center.x - tw / 2, center.y + th / 2, origin, ps, height)
            c1, r1 = world_to_pixel(center.x + tw / 2, center.y - th / 2, origin, ps, height)
            box = [c0, r0, c1, r1]
            draw.rectangle(box, fill=rgb)
            id_draws[aid].rectangle(box, fill=255)

        # obstacles
        for ob in state.obstacles:
            if ob.polyline and len(ob.polyline) >= 2:
                pix = [world_to_pixel(p.x, p.y, origin, ps, height) for p in ob.polyline]
                if ob.fill_role == "loop" and len(pix) >= 3:
                    # silhouette outline as obstacle (not filled) — leaders may leave the part
                    obs_draw.line(pix, fill=255, width=4)
                    draw.line(pix, fill=hex_to_rgb(self.obstacle_reserved[0]), width=1)
                else:
                    obs_draw.line(pix, fill=255, width=3)

        legend = ColorLegend(
            background=self.background,
            obstacle_reserved=self.obstacle_reserved,
            entries={},
        )

        for lin in state.linears:
            from layout_agent.geometry_parts import linear_parts

            color = lin.color
            legend.entries[color] = ColorLegendEntry(id=lin.id, category="linear")
            parts = linear_parts(lin)
            for seg in parts.extensions:
                line([seg.a, seg.b], color, lin.id)
            for seg in parts.dim_lines:
                line([seg.a, seg.b], color, lin.id, 2)
            mid = parts.text.center
            tw = float(lin.text.get("width", max(6.0, abs(lin.value) * 0.08)))
            th = float(lin.text.get("height", 3.5))
            text_blob(mid, color, lin.id, tw, th)

        for dia in list(state.diameters) + list(state.radials):
            color = dia.color
            legend.entries[color] = ColorLegendEntry(id=dia.id, category=dia.category)
            if len(dia.path) >= 2:
                line(dia.path, color, dia.id, 2)
            end = dia.path[-1] if dia.path else dia.anchor
            tw = float(dia.text.get("width", 10))
            th = float(dia.text.get("height", 5))
            # 尺寸线：文字下方短横线
            y_under = end.y - th / 2 - 0.4
            line(
                [Point(x=end.x - tw / 2, y=y_under), Point(x=end.x + tw / 2, y=y_under)],
                color,
                dia.id,
                2,
            )
            text_blob(Point(x=end.x, y=end.y), color, dia.id, tw, th)

        # pixel conflicts from per-id masks (ICM composite alone cannot show overwrite collisions)
        ids = sorted(id_masks.keys())
        conflict_pairs: List[Tuple[str, str]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                # logical AND via pointwise min
                inter = ImageChops.multiply(id_masks[a], id_masks[b])
                if inter.getbbox() is not None:
                    # count lit pixels roughly
                    hist = inter.histogram()
                    lit = sum(hist[1:])
                    if lit >= 4:
                        conflict_pairs.append((a, b))

        out_instance.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_instance)
        if out_obstacle:
            out_obstacle.parent.mkdir(parents=True, exist_ok=True)
            obs_img.save(out_obstacle)

        # update state raster paths (relative preferred by caller)
        state.color_legend = legend
        state.raster = RasterMeta(
            instance_image=str(out_instance.as_posix()),
            obstacle_image=str(out_obstacle.as_posix()) if out_obstacle else None,
            width=width,
            height=height,
            origin=origin,
            pixel_size=ps,
            y_down_image=True,
            anti_alias=False,
        )
        state.meta["revision"] = int(state.meta.get("revision", 0)) + 1
        state.meta["render_conflicts"] = conflict_pairs
        state.created_at = datetime.now(timezone.utc).isoformat()
        return state

    def save_state(self, state: LayoutState, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
