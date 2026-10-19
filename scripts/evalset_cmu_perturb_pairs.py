#!/usr/bin/env python3
"""CMU-style complex perturbation of perfect sides.

Inspired by Data-Augmentation-Engineering-Drawing (bound ± pad, side flip) but
never calls draw_random_dimension: p1/p2/anchor/value stay locked.

  data/evalsys/human_pairs/cmu_perturb.jsonl
  data/evalsys/cmu_perturb/states/
  data/evalsys/cmu_perturb/meta.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.evalset.human_gold import read_jsonl
from layout_agent.evalset.isolation import forbidden_training_views, jsonl_view_keys
from layout_agent.evalset.layout_diff import annotation_count, pose_delta
from layout_agent.evalset.perturb import (
    max_pose_shift_mm,
    measurement_signature,
    perturb_state_cmu_complex,
    perturb_state_dense,
    placement_outside_frame,
    text_leader_hits,
)
from layout_agent.evalset.preview import render_preview
from layout_agent.models import LayoutState

HP = ROOT / "data" / "evalsys" / "human_pairs"
OUT_DS = ROOT / "data" / "evalsys" / "cmu_perturb"
JSONL = HP / "cmu_perturb.jsonl"


def _load_state(rel: str) -> LayoutState:
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / rel
    return LayoutState.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()).as_posix())


def _skip_views(root: Path) -> Set[str]:
    keys = set(forbidden_training_views(root))
    keys |= jsonl_view_keys(HP / "test.jsonl")
    keys |= jsonl_view_keys(HP / "cross_view.jsonl")
    keys |= jsonl_view_keys(HP / "dropped.jsonl")
    return {k for k in keys if k}


def _complexity_candidates(
    rows: List[Dict[str, Any]],
    *,
    blocked: Set[str],
    min_ann: int,
    cap: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Prefer more-annotated views, then randomly order enough of them for 1:1."""
    usable: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for rec in rows:
        vkey = str(rec.get("view_key") or "")
        if not vkey or vkey in blocked or vkey in seen:
            continue
        if not rec.get("state_b"):
            continue
        seen.add(vkey)
        usable.append(rec)
    if not usable:
        return []
    complex_rows = [r for r in usable if int(r.get("n_ann") or 0) >= min_ann]
    rest = [r for r in usable if int(r.get("n_ann") or 0) < min_ann]
    rest.sort(key=lambda r: (-int(r.get("n_ann") or 0), str(r.get("pair_id") or "")))
    pool = list(complex_rows)
    if len(pool) < cap:
        pool.extend(rest[: max(0, cap - len(pool))])
    rng.shuffle(pool)
    return pool


_PERTURB_FNS = {
    "cmu": perturb_state_cmu_complex,
    "dense": perturb_state_dense,
}


def _try_perturb(
    rec: Dict[str, Any],
    rng: random.Random,
    *,
    min_delta: int,
    pass_id: str,
    states_dir: Path,
    rend_dir: Path,
    render: bool,
    method: str = "cmu",
) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        perfect = _load_state(str(rec["state_b"]))
    except Exception as exc:  # noqa: BLE001
        return None, f"parse:{exc}"
    if not (perfect.linears or perfect.diameters or perfect.radials):
        return None, "empty"
    sig = measurement_signature(perfect)
    perturb_fn = _PERTURB_FNS[method]
    best: Optional[Tuple[int, Any, List[str]]] = None
    attempts = 10 if method == "dense" else 6
    for _attempt in range(attempts):
        pert, ops = perturb_fn(perfect, rng)
        if not ops:
            continue
        if measurement_signature(pert) != sig:
            continue
        leaked = placement_outside_frame(pert)
        if leaked:
            continue
        delta = pose_delta(perfect, pert)
        if delta < min_delta:
            continue
        if best is None or delta > best[0]:
            best = (delta, pert, ops)
            if delta >= max(min_delta, 4):
                break
    if best is None:
        return None, "no_valid_perturb"
    delta, pert, ops = best
    src_pid = str(rec["pair_id"])
    pair_id = f"{pass_id}_{src_pid}"
    pert.state_id = f"{perfect.state_id or src_pid}_{pass_id}"
    src = dict(pert.source or {})
    src["cmu_from_pair"] = src_pid
    pert.source = {str(k): str(v) for k, v in src.items()}
    path_a = states_dir / f"{pair_id}_a.state.json"
    path_b = states_dir / f"{pair_id}_b.state.json"
    path_a.write_text(json.dumps(pert.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")
    path_b.write_text(json.dumps(perfect.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")
    if render:
        render_preview(pert, rend_dir / f"{pair_id}_a.png", max_px=1100)
        render_preview(perfect, rend_dir / f"{pair_id}_b.png", max_px=1100)
    vkey = str(rec["view_key"])
    row = {
        "pair_id": pair_id,
        "view_key": vkey,
        "source": "cmu_perturb_perfect",
        "src_pair_id": src_pid,
        "src_source": rec.get("source"),
        "solution_id": rec.get("solution_id"),
        "tenant_id": rec.get("tenant_id"),
        "dataset": str(Path(states_dir).parent.resolve().relative_to(ROOT.resolve()).as_posix()),
        "state_a": _rel(path_a),
        "state_b": _rel(path_b),
        "label": {"prefer": "b"},
        "view_type": rec.get("view_type"),
        "pose_delta": delta,
        "n_ann": rec.get("n_ann") or annotation_count(perfect),
        "quality": rec.get("quality"),
        "split": "cmu_perturb",
        "drawing_url": rec.get("drawing_url"),
        "perturbed_eid": rec.get("perfect_eid"),
        "perfect_eid": rec.get("perfect_eid"),
        "ops": ops,
        "synthetic": True,
        "synth_kind": "cmu_dense" if method == "dense" else "cmu_perturb",
        "n_ops": len(ops),
        "max_shift_mm": round(max_pose_shift_mm(perfect, pert), 2),
        "text_leader_hits": text_leader_hits(pert),
        "text_leader_hits_src": text_leader_hits(perfect),
    }
    return row, "ok"


def generate_cmu_pairs(
    source_rows: List[Dict[str, Any]],
    *,
    cap: int,
    seed: int,
    min_ann: int = 4,
    min_delta: int = 1,
    blocked: Optional[Set[str]] = None,
    render: bool = False,
    write_files: bool = True,
    method: str = "cmu",
    out_ds: Optional[Path] = None,
    jsonl_path: Optional[Path] = None,
    pass_prefix: str = "cmu",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Same-feature placement perturbation of perfect sides; count ≈ ``cap``."""
    if method not in _PERTURB_FNS:
        raise ValueError(f"unknown method {method}")
    blocked = {k for k in (blocked or set()) if k}
    ds = Path(out_ds) if out_ds else OUT_DS
    jsonl = Path(jsonl_path) if jsonl_path else JSONL
    states_dir = ds / "states"
    rend_dir = ds / "renders"
    states_dir.mkdir(parents=True, exist_ok=True)
    rend_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    candidates = _complexity_candidates(
        source_rows, blocked=blocked, min_ann=min_ann, cap=max(cap, len(source_rows)), rng=rng
    )
    kept: List[Dict[str, Any]] = []
    skips: Dict[str, int] = {}
    used_views: Set[str] = set()

    def absorb(rec: Dict[str, Any], pass_id: str) -> bool:
        if len(kept) >= cap:
            return False
        row, reason = _try_perturb(
            rec,
            rng,
            min_delta=min_delta,
            pass_id=pass_id,
            states_dir=states_dir,
            rend_dir=rend_dir,
            render=render,
            method=method,
        )
        if row is None:
            skips[reason.split(":")[0]] = skips.get(reason.split(":")[0], 0) + 1
            return False
        kept.append(row)
        used_views.add(str(rec["view_key"]))
        print(
            f"kept {row['pair_id']} n_ann={row.get('n_ann')} delta={row.get('pose_delta')} "
            f"ops={len(row.get('ops') or [])} shift={row.get('max_shift_mm')} "
            f"tl={row.get('text_leader_hits')}/{row.get('text_leader_hits_src')}"
        )
        return True

    for rec in candidates:
        if len(kept) >= cap:
            break
        absorb(rec, pass_prefix)

    if len(kept) < cap:
        for rec in candidates:
            if len(kept) >= cap:
                break
            vkey = str(rec.get("view_key") or "")
            if vkey in used_views:
                continue
            absorb(rec, pass_prefix)

    if len(kept) < cap:
        extra = [r for r in candidates if str(r.get("view_key") or "") in used_views]
        extra.sort(key=lambda r: (-int(r.get("n_ann") or 0), str(r.get("pair_id") or "")))
        for rec in extra:
            if len(kept) >= cap:
                break
            absorb(rec, f"{pass_prefix}2")

    if len(kept) < cap:
        extra = [r for r in candidates if str(r.get("view_key") or "") in used_views]
        extra.sort(key=lambda r: (-int(r.get("n_ann") or 0), str(r.get("pair_id") or "")))
        for rec in extra:
            if len(kept) >= cap:
                break
            absorb(rec, f"{pass_prefix}3")

    meta = {
        "seed": seed,
        "cap": cap,
        "min_ann": min_ann,
        "min_delta": min_delta,
        "measurement_lock": "p1/p2/anchor/value",
        "method": method,
        "frame_clamped": True,
        "forbidden_draw_random_dimension": True,
        "candidates": len(candidates),
        "kept": len(kept),
        "skips": skips,
        "jsonl": _rel(jsonl),
        "max_shift_mm_cap": 13.0 if method == "dense" else None,
        "text_on_dimline": method == "dense",
    }
    if write_files:
        jsonl.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
            encoding="utf-8",
        )
        (ds / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return kept, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--cap", type=int, default=0, help="0 = one synth per source row")
    ap.add_argument("--min-ann", type=int, default=4)
    ap.add_argument("--min-delta", type=int, default=1)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--pool", default=str(HP / "train.jsonl"))
    ap.add_argument("--method", choices=sorted(_PERTURB_FNS), default="cmu")
    ap.add_argument("--out-ds", default="")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--pass-prefix", default="")
    args = ap.parse_args()

    pool_path = Path(args.pool)
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    if not pool_path.exists():
        raise SystemExit(f"missing {pool_path}")
    rows = read_jsonl(pool_path)
    cap = args.cap if args.cap > 0 else len(rows)
    blocked = _skip_views(ROOT)
    prefix = args.pass_prefix or ("dense" if args.method == "dense" else "cmu")
    out_ds = Path(args.out_ds) if args.out_ds else None
    jsonl_path = Path(args.jsonl) if args.jsonl else None
    if out_ds is not None and not out_ds.is_absolute():
        out_ds = ROOT / out_ds
    if jsonl_path is not None and not jsonl_path.is_absolute():
        jsonl_path = ROOT / jsonl_path
    kept, meta = generate_cmu_pairs(
        rows,
        cap=cap,
        seed=args.seed,
        min_ann=args.min_ann,
        min_delta=args.min_delta,
        blocked=blocked,
        render=not args.no_render,
        method=args.method,
        out_ds=out_ds,
        jsonl_path=jsonl_path,
        pass_prefix=prefix,
    )
    print(
        f"\nwritten {len(kept)} pairs -> {meta['jsonl']}\n"
        f"candidates={meta['candidates']} skips={meta.get('skips')}"
    )


if __name__ == "__main__":
    main()
