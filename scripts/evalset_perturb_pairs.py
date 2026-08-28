#!/usr/bin/env python3
"""Generate R2 perturbation preference pairs from existing LayoutState snapshots.

For each source view: (original = prefer) vs (graded random perturbation = worse).
Also evaluates both sides with the formula Evaluator and reports how often the
perturbed side is indeed worse (sanity check of the weak-supervision assumption).

Usage:
  python scripts/evalset_perturb_pairs.py \
      --data-root data/datasets --out data/evalsys/pairs_perturb \
      --per-state 2 --levels light,medium,heavy --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.config import load_config
from layout_agent.evaluate import Evaluator
from layout_agent.evalset import LEVELS, PairWriter, make_pair_record, perturb_state, render_preview, view_key
from layout_agent.models import LayoutState
from layout_agent.perceive import Perceiver


def discover_states(data_root: Path) -> List[Path]:
    """All snapshot states, deduped by view_key (first occurrence wins)."""
    paths = sorted(data_root.glob("*/snapshots/*.state.json"))
    return paths


def load_state(path: Path) -> LayoutState:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return LayoutState.model_validate(raw)


def adaptive_pixel_size(state: LayoutState, base_ps: float, max_px: int = 2048) -> float:
    bbox = state.view.get("bbox") or {}
    bmin = bbox.get("min", {"x": 0, "y": 0})
    bmax = bbox.get("max", {"x": 100, "y": 80})
    span = max(
        float(bmax["x"]) - float(bmin["x"]),
        float(bmax["y"]) - float(bmin["y"]),
        1e-6,
    )
    return max(base_ps, span / max_px)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/datasets")
    ap.add_argument("--out", default="data/evalsys/pairs_perturb")
    ap.add_argument("--per-state", type=int, default=2, help="pairs per state per level")
    ap.add_argument("--levels", default="light,medium,heavy")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="cap number of source states (0=all)")
    ap.add_argument(
        "--min-dp",
        type=float,
        default=0.0,
        help="drop ambiguous pairs where P(perturbed)-P(original) < min_dp "
        "(sources are production snapshots, not expert finals, so a perturbation "
        "can accidentally improve the layout; such pairs are unreliable labels)",
    )
    args = ap.parse_args()

    data_root = (ROOT / args.data_root) if not Path(args.data_root).is_absolute() else Path(args.data_root)
    out_root = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    levels = [x.strip() for x in args.levels.split(",") if x.strip() in LEVELS]
    rng = random.Random(args.seed)

    cfg = load_config(ROOT / "config" / "default.yaml")
    perceiver = Perceiver(cfg)
    base_ps = perceiver.pixel_size
    evaluator = Evaluator(cfg)
    writer = PairWriter(out_root)

    seen_keys: set = set()
    stats: Dict[str, List[float]] = {lv: [] for lv in levels}
    n_pairs = 0
    n_states = 0
    n_skipped = 0
    n_ambiguous = 0

    states = discover_states(data_root)
    if args.limit:
        states = states[: args.limit]
    print(f"found {len(states)} snapshot states under {data_root}")

    for sp in states:
        try:
            state = load_state(sp)
        except Exception as e:  # noqa: BLE001
            print(f"skip {sp.name}: parse error {e}")
            n_skipped += 1
            continue
        vkey = view_key(state)
        if vkey in seen_keys:
            n_skipped += 1
            continue
        seen_keys.add(vkey)
        if not (state.linears or state.diameters or state.radials):
            n_skipped += 1
            continue

        perceiver.pixel_size = adaptive_pixel_size(state, base_ps)
        n_states += 1
        safe = state.state_id or f"s{n_states:04d}"

        # render + evaluate the original once per state
        try:
            pid0 = f"{safe}_orig"
            state_a = perceiver.render_icm(
                state,
                writer.icm / f"{pid0}.instance.png",
                writer.icm / f"{pid0}.obstacle.png",
            )
            rep_a = evaluator.evaluate(state_a)
        except Exception as e:  # noqa: BLE001
            print(f"skip {sp.name}: render/eval error {e}")
            n_skipped += 1
            continue

        for lv in levels:
            for k in range(args.per_state):
                pair_id = f"{safe}_{lv}{k}"
                try:
                    pert, ops = perturb_state(state, lv, rng)
                    if not ops:
                        continue
                    pert = perceiver.render_icm(
                        pert,
                        writer.icm_instance_path(pair_id, "b"),
                        writer.icm_obstacle_path(pair_id, "b"),
                    )
                    rep_b = evaluator.evaluate(pert)

                    # side a reuses the per-state original icm
                    writer.save_state(state_a, pair_id, "a")
                    writer.save_state(pert, pair_id, "b")
                    render_preview(state_a, writer.render_path(pair_id, "a"))
                    render_preview(pert, writer.render_path(pair_id, "b"))

                    p_a = float(rep_a.metrics.get("P", 0.0))
                    p_b = float(rep_b.metrics.get("P", 0.0))
                    stats[lv].append(p_b - p_a)
                    if p_b - p_a < args.min_dp - 1e-9:
                        n_ambiguous += 1
                        continue
                    rec = make_pair_record(
                        pair_id=pair_id,
                        vkey=vkey,
                        writer=writer,
                        source="perturb",
                        level=lv,
                        ops=ops,
                        prefer="a",
                        meta={
                            "P_a": p_a,
                            "P_b": p_b,
                            "V_hard_a": float(rep_a.metrics.get("V_hard", 0.0)),
                            "V_hard_b": float(rep_b.metrics.get("V_hard", 0.0)),
                            "src_state": str(sp.relative_to(data_root).as_posix()),
                        },
                    )
                    # fix icm_a to point at the shared original icm
                    rec["icm_a"] = f"icm/{pid0}.instance.png"
                    writer.append(rec)
                    n_pairs += 1
                except Exception as e:  # noqa: BLE001
                    print(f"pair {pair_id} failed: {e}")

    print(
        f"\nstates used: {n_states}, skipped: {n_skipped}, "
        f"pairs written: {n_pairs}, ambiguous dropped: {n_ambiguous}"
    )
    print(f"manifest: {writer.manifest}")
    print("\nassumption check (perturbed should be worse => dP > 0):")
    for lv, ds in stats.items():
        if not ds:
            continue
        worse = sum(1 for d in ds if d > 1e-9)
        equal = sum(1 for d in ds if abs(d) <= 1e-9)
        print(
            f"  {lv:6s}: n={len(ds)}, worse={worse} ({100*worse/len(ds):.1f}%), "
            f"equal={equal}, better={len(ds)-worse-equal}, avg dP={sum(ds)/len(ds):.1f}"
        )


if __name__ == "__main__":
    main()
