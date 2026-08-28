"""Preference-pair dataset IO for the learned evaluation dimension.

Pair schema follows《Agent效果优化-技术方案与数据需求》§3.2/§4.1:
one JSONL record per pair, states stored as separate files, plus
human-readable preview renders and ICM images for model input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from layout_agent.models import LayoutState


def view_key(state: LayoutState) -> str:
    """Stable per-view key: solution_id/view_uuid, fallback state_id."""
    src = state.source or {}
    sol = src.get("solution_id") or src.get("drawing_id") or ""
    vu = src.get("view_uuid") or src.get("view_id") or ""
    if sol and vu:
        return f"{sol}/{vu}"
    return state.state_id or "unknown"


def split_for(key: str, val_ratio: float = 0.15) -> str:
    """Deterministic train/val split by view_key hash (same view never crosses sets)."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    return "val" if (h % 1000) < int(val_ratio * 1000) else "train"


class PairWriter:
    """Writes pairs under out_root:

    out_root/
      manifest.jsonl   # one pair per line
      states/          # {pair_id}_{a|b}.state.json
      renders/         # {pair_id}_{a|b}.png        (human preview)
      icm/             # {pair_id}_{a|b}.instance.png / .obstacle.png
    """

    def __init__(self, out_root: Path):
        self.root = Path(out_root)
        self.states = self.root / "states"
        self.renders = self.root / "renders"
        self.icm = self.root / "icm"
        for d in (self.states, self.renders, self.icm):
            d.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.jsonl"

    def state_path(self, pair_id: str, which: str) -> Path:
        return self.states / f"{pair_id}_{which}.state.json"

    def render_path(self, pair_id: str, which: str) -> Path:
        return self.renders / f"{pair_id}_{which}.png"

    def icm_instance_path(self, pair_id: str, which: str) -> Path:
        return self.icm / f"{pair_id}_{which}.instance.png"

    def icm_obstacle_path(self, pair_id: str, which: str) -> Path:
        return self.icm / f"{pair_id}_{which}.obstacle.png"

    def save_state(self, state: LayoutState, pair_id: str, which: str) -> Path:
        p = self.state_path(pair_id, which)
        raw = state.model_dump(mode="json")
        # store icm paths relative to out_root for portability
        raster = raw.get("raster") or {}
        for k in ("instance_image", "obstacle_image"):
            v = raster.get(k)
            if v:
                try:
                    raster[k] = str(Path(v).resolve().relative_to(self.root.resolve()).as_posix())
                except ValueError:
                    pass
        p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return p

    def append(self, record: Dict[str, Any]) -> None:
        with open(self.manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_pair_record(
    *,
    pair_id: str,
    vkey: str,
    writer: PairWriter,
    source: str,
    level: Optional[str] = None,
    ops: Optional[list] = None,
    prefer: str = "a",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rel = lambda p: str(p.relative_to(writer.root).as_posix())  # noqa: E731
    rec: Dict[str, Any] = {
        "pair_id": pair_id,
        "view_key": vkey,
        "split": split_for(vkey),
        "state_a": rel(writer.state_path(pair_id, "a")),
        "state_b": rel(writer.state_path(pair_id, "b")),
        "render_a": rel(writer.render_path(pair_id, "a")),
        "render_b": rel(writer.render_path(pair_id, "b")),
        "icm_a": rel(writer.icm_instance_path(pair_id, "a")),
        "icm_b": rel(writer.icm_instance_path(pair_id, "b")),
        "label": {"prefer": prefer},
        "source": source,
    }
    if level:
        rec["level"] = level
    if ops:
        rec["ops"] = ops
    if meta:
        rec["meta"] = meta
    return rec
