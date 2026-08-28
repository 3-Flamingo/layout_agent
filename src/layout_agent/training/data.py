"""Dataset for pairwise reward-model training.

Input encoding (C x H x W, float32 in [0,1]):
  ch0: annotation occupancy  (ICM instance image, any non-background pixel)
  ch1: obstacle lines        (ICM obstacle mask)
  ch2: text boxes            (rasterized from the LayoutState text AABBs)

Images are letterboxed (aspect-preserving) to img_size x img_size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from layout_agent.geometry_parts import collect_parts
from layout_agent.models import LayoutState
from layout_agent.perceive import world_to_pixel


def _letterbox(arr: np.ndarray, size: int) -> np.ndarray:
    """Aspect-preserving resize of HxW uint8 mask into size x size (nearest)."""
    h, w = arr.shape
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    img = Image.fromarray(arr).resize((nw, nh), Image.NEAREST)
    out = np.zeros((size, size), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top : top + nh, left : left + nw] = np.asarray(img)
    return out


def _text_mask(state: LayoutState) -> np.ndarray:
    """Rasterize text AABBs into the state's own raster grid (uint8 0/255)."""
    r = state.raster
    mask = np.zeros((r.height, r.width), dtype=np.uint8)
    for p in collect_parts(state).values():
        t = p.text
        c0, r0 = world_to_pixel(t.xmin, t.ymax, r.origin, r.pixel_size, r.height)
        c1, r1 = world_to_pixel(t.xmax, t.ymin, r.origin, r.pixel_size, r.height)
        c0, c1 = max(0, min(c0, c1)), min(r.width - 1, max(c0, c1))
        r0, r1 = max(0, min(r0, r1)), min(r.height - 1, max(r0, r1))
        if c1 >= c0 and r1 >= r0:
            mask[r0 : r1 + 1, c0 : c1 + 1] = 255
    return mask


def encode_channels(
    instance_png: Path,
    obstacle_png: Optional[Path],
    state: Optional[LayoutState],
    img_size: int,
    channels: int = 3,
) -> np.ndarray:
    """Return uint8 array [C, img_size, img_size] with values 0/255."""
    inst = np.asarray(Image.open(instance_png).convert("RGB"))
    occ = (inst.max(axis=2) > 0).astype(np.uint8) * 255
    chans = [_letterbox(occ, img_size)]
    if channels >= 2:
        if obstacle_png is not None and Path(obstacle_png).exists():
            obs = np.asarray(Image.open(obstacle_png).convert("L"))
            obs = (obs > 0).astype(np.uint8) * 255
        else:
            obs = np.zeros_like(occ)
        chans.append(_letterbox(obs, img_size))
    if channels >= 3:
        if state is not None:
            chans.append(_letterbox(_text_mask(state), img_size))
        else:
            chans.append(np.zeros((img_size, img_size), dtype=np.uint8))
    return np.stack(chans, axis=0)


def _obstacle_for(instance_path: Path) -> Path:
    return Path(str(instance_path).replace(".instance.png", ".obstacle.png"))


class StateEncoder:
    """Encode a LayoutState (with existing or freshly rendered ICM) to a tensor.

    Used by eval.py and env.py; training reads pre-rendered pair files instead.
    """

    def __init__(self, img_size: int = 256, channels: int = 3, work_dir: Optional[Path] = None):
        self.img_size = img_size
        self.channels = channels
        self.work = work_dir
        self._perceiver = None

    def encode(self, state: LayoutState, base_dir: Optional[Path] = None) -> np.ndarray:
        inst = Path(state.raster.instance_image or "")
        if not inst.is_absolute() and base_dir is not None:
            inst = base_dir / inst
        if not inst.exists():
            from layout_agent.config import load_config
            from layout_agent.perceive import Perceiver

            if self._perceiver is None:
                root = Path(__file__).resolve().parents[3]
                self._perceiver = Perceiver(load_config(root / "config" / "default.yaml"))
            work = self.work or Path.cwd() / "_encode_work"
            work.mkdir(parents=True, exist_ok=True)
            sid = state.state_id or "v"
            state = self._perceiver.render_icm(
                state, work / f"{sid}.instance.png", work / f"{sid}.obstacle.png"
            )
            inst = Path(state.raster.instance_image)
        obs = Path(state.raster.obstacle_image) if state.raster.obstacle_image else _obstacle_for(inst)
        if not obs.is_absolute() and base_dir is not None:
            obs = base_dir / obs
        return encode_channels(inst, obs, state, self.img_size, self.channels)


class PairDataset:
    """(better, worse) image pairs from an evalset manifest.jsonl.

    Yields dicts with float32 tensors "pos" / "neg" (better first) plus meta.
    Torch-free __getitem__ output (numpy) so the module imports without torch;
    train.py wraps it in a DataLoader with a collate that converts to tensors.
    """

    def __init__(
        self,
        manifest: Path,
        split: str,
        img_size: int = 256,
        channels: int = 3,
        cache: bool = True,
    ):
        self.root = Path(manifest).parent
        self.img_size = img_size
        self.channels = channels
        self.records: List[Dict] = []
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            prefer = str(rec.get("label", {}).get("prefer", "")).lower()
            if prefer not in ("a", "b"):
                continue
            if split and rec.get("split") != split:
                continue
            self.records.append(rec)
        self._cache: Optional[Dict[str, np.ndarray]] = {} if cache else None

    def __len__(self) -> int:
        return len(self.records)

    def _encode_side(self, rec: Dict, which: str) -> np.ndarray:
        icm_key = f"icm_{which}"
        state_key = f"state_{which}"
        cache_key = rec.get(icm_key, "") + f"|{which}|{rec['pair_id']}"
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]
        inst = self.root / rec[icm_key]
        state = None
        if self.channels >= 3 and rec.get(state_key):
            sp = self.root / rec[state_key]
            if sp.exists():
                state = LayoutState.model_validate(json.loads(sp.read_text(encoding="utf-8")))
        arr = encode_channels(inst, _obstacle_for(inst), state, self.img_size, self.channels)
        if self._cache is not None:
            self._cache[cache_key] = arr
        return arr

    def __getitem__(self, i: int) -> Dict:
        rec = self.records[i]
        prefer = rec["label"]["prefer"]
        pos_side, neg_side = ("a", "b") if prefer == "a" else ("b", "a")
        pos = self._encode_side(rec, pos_side).astype(np.float32) / 255.0
        neg = self._encode_side(rec, neg_side).astype(np.float32) / 255.0
        return {
            "pos": pos,
            "neg": neg,
            "level": rec.get("level", ""),
            "pair_id": rec["pair_id"],
        }


def load_state_file(path: Path) -> Tuple[LayoutState, Path]:
    """Load a state json; return (state, base_dir for relative raster paths)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    st = LayoutState.model_validate(raw)
    return st, Path(path).parent.parent  # states/xx.json -> pairs root
