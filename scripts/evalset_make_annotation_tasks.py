#!/usr/bin/env python3
"""Build an annotation package (tasks.js + annotate.html + renders).

Two modes:
  prefer: sample pairs from a pairs manifest.jsonl (T2 / R3 labeling)
  grade:  render previews for a set of LayoutState files (T1 labeling)

The output folder is self-contained: open annotate.html in a browser,
label, then export labels .jsonl (format matches evalset_baseline_consistency).

Usage:
  python scripts/evalset_capture_official.py --from-t2 --from-t1
  python scripts/evalset_make_annotation_tasks.py prefer \
      --manifest data/evalsys/pairs_perturb/manifest.jsonl \
      --split val --out data/evalsys/annotate_t2 --renderer official
  python scripts/evalset_make_annotation_tasks.py grade \
      --states "data/datasets/*/snapshots/*.state.json" \
      --holdout data/evalsys/holdout_views.txt \
      --out data/evalsys/annotate_t1 --renderer official
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.evalset import render_preview, view_key
from layout_agent.models import LayoutState


def _state_key(path: Path) -> str:
    name = path.name
    if name.endswith(".state.json"):
        return name[: -len(".state.json")]
    return path.stem


def _add_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--renderer",
        choices=("preview", "official"),
        default="official",
        help="official = CadView 投图截图（需先跑 evalset_capture_official.py）",
    )
    p.add_argument(
        "--official-dir",
        default="data/evalsys/official_renders",
        help="PNG dir written by evalset_capture_official.py, keyed by state file stem",
    )


def _place_render(st: LayoutState, state_path: Path, dst: Path, renderer: str, official_dir: Path) -> None:
    if renderer == "official":
        src = official_dir / f"{_state_key(state_path)}.png"
        if not src.exists():
            raise FileNotFoundError(
                f"缺少官网投图截图 {src}。请先运行 scripts/evalset_capture_official.py"
            )
        shutil.copy(src, dst)
        return
    render_preview(st, dst)


def _load_holdout(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    p = Path(path)
    keys = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")}
    print(f"holdout: {len(keys)} view_keys from {p}")
    return keys


PUBLIC_SOLUTION_IDS = {"2082016429787918336"}
PUBLIC_DRAWING_HOST = "https://drawing.designorder.cn"
ZHONGCHE_DRAWING_HOST = "https://v3.private.designorder.cn/drawing"
_TREE_LOOKUP: Optional[Dict[str, Dict[str, str]]] = None


def _eid(obj: object) -> Optional[int]:
    if obj is None:
        return None
    if isinstance(obj, dict) and "id" in obj:
        return int(obj["id"])  # type: ignore[arg-type]
    if isinstance(obj, (int, float)):
        return int(obj)
    return None


def load_view_tree_lookup() -> Dict[str, Dict[str, str]]:
    """CadView.uuid (short) → CadTreeNode GUID + part name, from OSS dumps."""
    global _TREE_LOOKUP
    if _TREE_LOOKUP is not None:
        return _TREE_LOOKUP
    lookup: Dict[str, Dict[str, str]] = {}
    for p in sorted((ROOT / "data" / "raw").glob("solution_*/solution_doc.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        by: Dict[str, List[Dict]] = {}
        for el in doc.get("doc") or []:
            by.setdefault(str(el.get("_ctor_") or "Unknown"), []).append(el)
        nodes = {_eid(n.get("id")): n for n in by.get("CadTreeNode", [])}
        groups = {_eid(g.get("id")): g for g in by.get("CadViewGroup", [])}
        sid = p.parent.name.replace("solution_", "")
        for v in by.get("CadView", []):
            vu = str(v.get("uuid") or "")
            g = groups.get(_eid(v.get("pid")))
            if not g or not vu:
                continue
            n = nodes.get(_eid(g.get("treeNodeId")))
            if not n:
                continue
            lookup[f"{sid}/{vu}"] = {
                "tree_uuid": str(n.get("uuid") or ""),
                "part_name": str(n.get("name") or n.get("pureName") or ""),
                "view_type": str(v.get("viewType") or ""),
                "view_uuid": vu,
            }
    _TREE_LOOKUP = lookup
    return lookup


def _enrich_task(task: Dict, vkey: str) -> None:
    meta = load_view_tree_lookup().get(vkey) or {}
    for k in ("tree_uuid", "part_name", "view_type", "view_uuid"):
        if meta.get(k):
            task[k] = meta[k]


def _is_live_solution_id(sid: str) -> bool:
    return bool(sid.isdigit() and len(sid) >= 10)


def _official_drawing_url(st: LayoutState) -> str:
    src = st.source or {}
    url = str(src.get("drawing_url") or "").strip()
    if url.startswith("http"):
        return url
    sid = str(src.get("solution_id") or "")
    if not _is_live_solution_id(sid):
        return ""
    tenant = str(src.get("tenant_id") or "")
    if sid in PUBLIC_SOLUTION_IDS or tenant == "0":
        return f"{PUBLIC_DRAWING_HOST}/?solutionId={sid}&tenantId=0"
    return f"{ZHONGCHE_DRAWING_HOST}/?solutionId={sid}&tenantId={tenant or '250'}"


def _layout_fingerprint(st: LayoutState) -> str:
    """Cheap identity so original vs each perturbation stay distinct T1 items."""
    bits: List[str] = []
    for a in st.linears:
        tp = a.text_point
        tps = f"{tp.x:.2f}:{tp.y:.2f}" if tp else "none"
        bits.append(f"L:{a.id}:{a.point.x:.2f}:{a.point.y:.2f}:{tps}")
    for a in list(st.diameters) + list(st.radials):
        last = a.path[-1] if a.path else a.anchor
        bits.append(f"R:{a.id}:{last.x:.2f}:{last.y:.2f}")
    return "|".join(sorted(bits)) if bits else st.state_id or "empty"


def _write_package(out: Path, tasks: List[Dict], dataset_id: str) -> None:
    ids = [str(t.get("id") or "") for t in tasks]
    duplicates = sorted(
        ident for ident, count in Counter(ids).items() if ident and count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate annotation task ids: {duplicates}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "tools" / "annotate.html", out / "annotate.html")
    js = (
        f"window.DATASET_ID = {json.dumps(dataset_id)};\n"
        f"window.TASKS = {json.dumps(tasks, ensure_ascii=False)};\n"
    )
    (out / "tasks.js").write_text(js, encoding="utf-8")
    print(f"package: {out}\\annotate.html  ({len(tasks)} tasks)")


def mode_prefer(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    pairs_root = manifest.parent
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    rng = random.Random(args.seed)

    records = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    if getattr(args, "split", None):
        records = [r for r in records if r.get("split") == args.split]
        print(f"filtered split={args.split!r}: {len(records)} pairs")
    holdout = _load_holdout(getattr(args, "holdout", None))
    if holdout is not None:
        records = [r for r in records if r.get("view_key") in holdout]
        print(f"filtered holdout: {len(records)} pairs")
    if args.n and len(records) > args.n:
        records = rng.sample(records, args.n)

    rend = out / "renders"
    rend.mkdir(parents=True, exist_ok=True)
    official_dir = Path(args.official_dir)
    if not official_dir.is_absolute():
        official_dir = ROOT / official_dir
    tasks = []
    for rec in records:
        pid = rec["pair_id"]
        # randomize left/right so annotators can't learn "a is always original"
        flip = rng.random() < 0.5
        ra, rb = rec["render_a"], rec["render_b"]
        sa, sb = rec["state_a"], rec["state_b"]
        if flip:
            ra, rb, sa, sb = rb, ra, sb, sa
        missing = False
        for src, dst, state_rel in ((ra, f"{pid}_L.png", sa), (rb, f"{pid}_R.png", sb)):
            dst_p = rend / dst
            sp = Path(state_rel)
            if not sp.is_absolute():
                sp = pairs_root / state_rel
            st = LayoutState.model_validate(json.loads(sp.read_text(encoding="utf-8")))
            if not _official_drawing_url(st):
                print(f"skip {pid}: no live 官网投图 (synthetic/oss)")
                missing = True
                break
            try:
                _place_render(st, sp, dst_p, args.renderer, official_dir)
            except FileNotFoundError as e:
                print(f"skip {pid} {dst}: {e}")
                missing = True
                break
        if missing:
            continue
        tasks.append(
            {
                "id": pid,
                "type": "prefer",
                "view_key": rec.get("view_key", ""),
                "drawing_url": _official_drawing_url(st),
                "render_a": f"renders/{pid}_L.png",
                "render_b": f"renders/{pid}_R.png",
                # state paths follow the displayed order so exported labels stay correct
                "state_a": str((pairs_root / sa).resolve().as_posix()),
                "state_b": str((pairs_root / sb).resolve().as_posix()),
            }
        )
        _enrich_task(tasks[-1], rec.get("view_key", ""))
    _write_package(out, tasks, out.name)


def mode_grade(args: argparse.Namespace) -> None:
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    rng = random.Random(args.seed)
    holdout = _load_holdout(getattr(args, "holdout", None))
    paths = [Path(p) for p in glob.glob(args.states, recursive=True)] if args.states else []
    if getattr(args, "from_pairs", None):
        print("note: --from-pairs ignored for T1 (synthetic perturb has no matching 官网原图)")
    # skip synthetic perturb states even if the glob hits evalsys/pairs_perturb
    paths = [p for p in paths if "pairs_perturb" not in str(p).replace("\\", "/")]
    # dedupe by resolved path, then by (view_key, layout fingerprint)
    seen_path: Set[str] = set()
    uniq: List[Path] = []
    seen_fp: Set[str] = set()
    for p in sorted(paths, key=lambda x: str(x)):
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen_path:
            continue
        seen_path.add(key)
        if not p.exists():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            st = LayoutState.model_validate(raw)
        except Exception:  # noqa: BLE001
            continue
        vk = view_key(st)
        if holdout is not None and vk not in holdout:
            continue
        fp = f"{vk}|{_layout_fingerprint(st)}"
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        uniq.append(p)
    if args.n and len(uniq) > args.n:
        uniq = rng.sample(uniq, args.n)

    rend = out / "renders"
    rend.mkdir(parents=True, exist_ok=True)
    official_dir = Path(args.official_dir)
    if not official_dir.is_absolute():
        official_dir = ROOT / official_dir
    tasks = []
    for p in uniq:
        st = LayoutState.model_validate(json.loads(p.read_text(encoding="utf-8")))
        if not _official_drawing_url(st):
            print(f"skip {p.name}: no live 官网投图 (synthetic/oss)")
            continue
        sid = p.stem
        try:
            _place_render(st, p, rend / f"{sid}.png", args.renderer, official_dir)
        except Exception as e:  # noqa: BLE001
            print(f"render failed {p.name}: {e}")
            continue
        tasks.append(
            {
                "id": sid,
                "type": "grade",
                "view_key": view_key(st),
                "drawing_url": _official_drawing_url(st),
                "render": f"renders/{sid}.png",
                "state": str(p.resolve().as_posix()),
            }
        )
        _enrich_task(tasks[-1], view_key(st))
        last = tasks[-1]
        if not (last.get("part_name") or last.get("tree_uuid")):
            print(f"skip {p.name}: no online part (cannot locate 官网单图)")
            tasks.pop()
    _write_package(out, tasks, out.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p1 = sub.add_parser("prefer")
    p1.add_argument("--manifest", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--n", type=int, default=0)
    p1.add_argument("--seed", type=int, default=7)
    p1.add_argument("--split", default="", help="keep only this split, e.g. val")
    p1.add_argument("--holdout", default="", help="txt file of view_keys, one per line")
    _add_render_args(p1)
    p2 = sub.add_parser("grade")
    p2.add_argument("--states", default="", help="glob, e.g. data/datasets/*/snapshots/*.state.json")
    p2.add_argument("--out", required=True)
    p2.add_argument("--n", type=int, default=0)
    p2.add_argument("--seed", type=int, default=7)
    p2.add_argument("--holdout", default="", help="txt file of view_keys, one per line")
    p2.add_argument("--split", default="", help="when used with --from-pairs, keep this split")
    p2.add_argument(
        "--from-pairs",
        default="",
        help="ignored for T1: synthetic perturb images do not match 官网原图",
    )
    _add_render_args(p2)
    args = ap.parse_args()
    if args.mode == "prefer":
        mode_prefer(args)
    else:
        mode_grade(args)


if __name__ == "__main__":
    main()
