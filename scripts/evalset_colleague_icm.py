#!/usr/bin/env python3
"""Run the colleague 100-pair set through the stored ICM pipeline.

1. Patch snapshots with CAD roughness + dim deviations (keyed by CadView id,
   because perturbed/perfect copies share a view uuid).
2. Render instance/obstacle PNGs next to the dataset (same adaptive pixel size
   as perturb-pair ICM).
3. Point colleague_challenge.jsonl at those PNGs.
4. Re-score the frozen reward checkpoint.

Does not overwrite the 2026-09-08 on-the-fly eval JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from fetch_production_dataset import (  # noqa: E402
    _eid,
    aligned_measurement,
    convert_roughness_block,
    estimate_text,
    format_dim_value,
)
from evalset_perturb_pairs import adaptive_pixel_size  # noqa: E402
from evalset_repair_tol_roughness import _v2, load_tasks  # noqa: E402
from layout_agent.config import load_config  # noqa: E402
from layout_agent.evalset import render_preview  # noqa: E402
from layout_agent.models import LayoutState, Point  # noqa: E402
from layout_agent.perceive import Perceiver  # noqa: E402

SID = "1537422555585998848"
DS = ROOT / "data" / "datasets" / "prod_colleague_pairs_598848"
CHALLENGE = ROOT / "data" / "evalsys" / "proxy" / "colleague_challenge.jsonl"
CKPT = ROOT / "runs" / "reward_proxy_a800_strict" / "best.pt"


def load_colleague_catalog(
    sid: str = SID,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    p = ROOT / "data" / "raw" / f"solution_{sid}" / "solution_doc.json"
    doc = json.loads(p.read_text(encoding="utf-8"))["doc"]
    views: Dict[Any, Dict[str, Any]] = {}
    by_pid: Dict[Any, List[Dict[str, Any]]] = {}
    for el in doc:
        ctor = el.get("_ctor_")
        if ctor == "CadView":
            views[_eid(el.get("id"))] = el
        pid = _eid(el.get("pid"))
        if pid is not None:
            by_pid.setdefault(pid, []).append(el)
    dims: Dict[Tuple[str, str], Dict[str, Any]] = {}
    roughness: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    n_rg = 0
    for vid, view in views.items():
        vmat = view.get("matrix")
        rgs: List[Dict[str, Any]] = []
        for el in by_pid.get(vid, []):
            ctor = el.get("_ctor_")
            uid = str(el.get("uuid") or "")
            ov = el.get("override") or {}
            if ctor == "CadAlignedDimension" and uid:
                text_ov = ov.get("text") if isinstance(ov.get("text"), dict) else {}
                try:
                    ang = float(el.get("dimAngle") if el.get("dimAngle") is not None else 0)
                except (TypeError, ValueError):
                    ang = 0.0
                dims[(sid, uid)] = {
                    "kind": "linear",
                    "override": ov,
                    "dim_angle": ang,
                    "dimdec": int(text_ov.get("dimdec") if text_ov.get("dimdec") is not None else 2),
                    "dimrnd": float(text_ov.get("dimrnd") or 0),
                    "measurementType": str(el.get("measurementType") or "world"),
                }
            elif ctor == "CadDiameterDimension" and uid:
                pu = ov.get("primaryUnit") if isinstance(ov.get("primaryUnit"), dict) else {}
                text_ov = ov.get("text") if isinstance(ov.get("text"), dict) else {}
                hole = el.get("relatedHoleDetailedInfo") or {}
                dims[(sid, uid)] = {
                    "kind": "diameter",
                    "override": ov,
                    "prefix": pu.get("dimPrefix") or "⌀",
                    "suffix": pu.get("dimSuffix") or "",
                    "dimdec": int(text_ov.get("dimdec") if text_ov.get("dimdec") is not None else 2),
                    "diameter": hole.get("diameter_value"),
                }
            elif ctor == "CadRadialDimension" and uid:
                pu = ov.get("primaryUnit") if isinstance(ov.get("primaryUnit"), dict) else {}
                text_ov = ov.get("text") if isinstance(ov.get("text"), dict) else {}
                cx, cy = _v2(el.get("center"))
                hx, hy = _v2(el.get("chordPoint"))
                dims[(sid, uid)] = {
                    "kind": "radial",
                    "override": ov,
                    "prefix": pu.get("dimPrefix") or "R",
                    "suffix": pu.get("dimSuffix") or "",
                    "dimdec": int(text_ov.get("dimdec") if text_ov.get("dimdec") is not None else 2),
                    "radius": math.hypot(hx - cx, hy - cy),
                }
            rg = convert_roughness_block(el, vmat)
            if rg is not None:
                rgs.append(rg.model_dump(mode="json"))
        roughness[(sid, str(vid))] = rgs
        n_rg += len(rgs)
    print(f"catalog {sid}: {len(dims)} dims, {n_rg} roughness on {len(views)} views")
    return dims, roughness


def patch_one(
    raw: Dict[str, Any],
    dims: Dict[Tuple[str, str], Dict[str, Any]],
    roughness: Dict[Tuple[str, str], List[Dict[str, Any]]],
) -> int:
    src = raw.get("source") or {}
    sid = str(src.get("solution_id") or "")
    vid = str(src.get("cad_view_id") or src.get("view_id") or "")
    n = 0
    for lin in raw.get("linears") or []:
        meta = dims.get((sid, str(lin.get("id") or "")))
        if not meta or meta.get("kind") != "linear":
            continue
        p1 = lin.get("p1") or {}
        p2 = lin.get("p2") or {}
        value = aligned_measurement(
            Point(x=float(p1.get("x", 0)), y=float(p1.get("y", 0))),
            Point(x=float(p2.get("x", 0)), y=float(p2.get("y", 0))),
            float(meta["dim_angle"]),
            measurement_type=str(meta.get("measurementType") or "world"),
            dimrnd=float(meta.get("dimrnd") or 0),
        )
        content = format_dim_value(value, int(meta["dimdec"]))
        dim_scale = float(lin.get("dim_scale") or 1.0)
        new_text = estimate_text(meta.get("override") or {}, content, dim_scale)
        old = lin.get("text") or {}
        lin["value"] = value
        lin["dim_angle"] = meta["dim_angle"]
        lin["text"] = new_text
        if old.get("content") != new_text.get("content") or old.get("tolerance") != new_text.get("tolerance"):
            n += 1
    for a in list(raw.get("diameters") or []) + list(raw.get("radials") or []):
        meta = dims.get((sid, str(a.get("id") or "")))
        if not meta or meta.get("kind") not in ("diameter", "radial"):
            continue
        if meta["kind"] == "diameter":
            diam = meta.get("diameter")
            if diam is None:
                content = f"{meta['prefix']}{meta['suffix']}" or "⌀"
            else:
                content = f"{meta['prefix']}{format_dim_value(float(diam), int(meta['dimdec']))}{meta['suffix']}"
        else:
            content = f"{meta['prefix']}{format_dim_value(float(meta['radius']), int(meta['dimdec']))}{meta['suffix']}"
        dim_scale = float(a.get("dim_scale") or 1.0)
        new_text = estimate_text(meta.get("override") or {}, content, dim_scale)
        old = a.get("text") or {}
        a["text"] = new_text
        if old.get("content") != new_text.get("content") or old.get("tolerance") != new_text.get("tolerance"):
            n += 1
    rgs = roughness.get((sid, vid))
    if rgs is not None:
        prev = raw.get("roughness") or []
        raw["roughness"] = rgs
        meta = dict(raw.get("meta") or {})
        meta["roughness_count"] = len(rgs)
        raw["meta"] = meta
        if prev != rgs:
            n += 1
    return n


def render_icm_dataset(states: List[Path], base_ps: float) -> None:
    cfg = load_config(ROOT / "config" / "default.yaml")
    perceiver = Perceiver(cfg)
    icm_dir = DS / "icm"
    icm_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    for p in states:
        raw = json.loads(p.read_text(encoding="utf-8"))
        st = LayoutState.model_validate(raw)
        sid = st.state_id or p.stem.replace(".state", "")
        rel_inst = f"icm/{sid}.instance.png"
        rel_obs = f"icm/{sid}.obstacle.png"
        perceiver.pixel_size = adaptive_pixel_size(st, base_ps)
        try:
            st = perceiver.render_icm(st, DS / rel_inst, DS / rel_obs)
        except MemoryError:
            perceiver.pixel_size = adaptive_pixel_size(st, base_ps, max_px=1024)
            try:
                st = perceiver.render_icm(st, DS / rel_inst, DS / rel_obs)
            except MemoryError:
                print(f"ICM MemoryError {p.name}")
                n_fail += 1
                continue
        st.raster.instance_image = rel_inst
        st.raster.obstacle_image = rel_obs
        p.write_text(json.dumps(st.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        n_ok += 1
        if n_ok % 20 == 0:
            print(f"  ICM {n_ok}/{len(states)}")
    print(f"ICM rendered {n_ok}, failed {n_fail}")


def update_challenge_jsonl(path: Path) -> None:
    rows = []
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for side in ("a", "b"):
            sp = rec.get(f"state_{side}") or ""
            stem = Path(sp).name.replace(".state.json", "")
            rec[f"icm_{side}"] = f"../../datasets/prod_colleague_pairs_598848/icm/{stem}.instance.png"
        rows.append(rec)
        n += 1
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"updated {n} challenge records with icm_a/icm_b -> {path}")


def rerender_colleague_previews() -> None:
    pack = ROOT / "data" / "evalsys" / "annotate_t2"
    tasks = load_tasks(pack / "tasks.js")
    rend = pack / "renders"
    n = 0
    for t in tasks:
        if t.get("source") != "colleague_pairs" and not str(t.get("id") or "").startswith("colleague_"):
            continue
        for sp, rp in ((t.get("state_a"), t.get("render_a")), (t.get("state_b"), t.get("render_b"))):
            if not sp or not rp:
                continue
            src = Path(sp)
            dst = rend / Path(rp).name
            if not src.exists():
                print(f"missing {src}")
                continue
            st = LayoutState.model_validate(json.loads(src.read_text(encoding="utf-8")))
            render_preview(st, dst)
            n += 1
    print(f"re-rendered {n} colleague T2 paper previews")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-preview", action="store_true")
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    dims, roughness = load_colleague_catalog()
    snaps = sorted((DS / "snapshots").glob("*.state.json"))
    n_files = n_fields = n_rg = n_tol = 0
    for p in snaps:
        raw = json.loads(p.read_text(encoding="utf-8"))
        n = patch_one(raw, dims, roughness)
        n_files += 1
        n_fields += n
        if raw.get("roughness"):
            n_rg += 1
        if any((x.get("text") or {}).get("tolerance") for x in (raw.get("linears") or [])):
            n_tol += 1
        p.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"patched {n_files} states ({n_fields} field updates); "
          f"{n_rg} with roughness, {n_tol} with dimtol")

    cfg = load_config(ROOT / "config" / "default.yaml")
    base_ps = float((cfg.get("perceive") or {}).get("pixel_size", 0.35))
    render_icm_dataset(snaps, base_ps)
    update_challenge_jsonl(CHALLENGE)
    if not args.skip_preview:
        rerender_colleague_previews()

    if args.skip_eval:
        return
    if not CKPT.exists():
        print(f"skip eval: missing {CKPT}")
        return
    from layout_agent.training.eval import CheckpointScorer, eval_t2  # noqa: E402

    device = args.device or None
    scorer = CheckpointScorer(CKPT, device)
    report = {
        "ckpt": str(CKPT.as_posix()),
        "split": "challenge",
        "icm": "stored",
        "t2": eval_t2(CHALLENGE, scorer, "challenge"),
    }
    out = ROOT / "runs" / "reward_proxy_a800_strict" / "eval_colleague_challenge_icm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
