#!/usr/bin/env python3
"""Append newly fetched live production views into annotate_t1 / annotate_t2.

Keeps existing task ids (already-labeled items stay valid).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evalset_make_annotation_tasks import (  # noqa: E402
    _enrich_task,
    _official_drawing_url,
    _write_package,
)
from layout_agent.evalset import render_preview  # noqa: E402
from layout_agent.models import LayoutState  # noqa: E402
from layout_agent.evalset.pairs import view_key  # noqa: E402

LIVE = re.compile(r"^(\d{10,})/")


def load_tasks(pack: Path) -> tuple[list, str]:
    text = (pack / "tasks.js").read_text(encoding="utf-8")
    ds = "default"
    m = re.search(r"window\.DATASET_ID\s*=\s*(.*?);", text)
    if m:
        ds = json.loads(m.group(1))
    i = text.find("window.TASKS = ")
    tasks = json.loads(text[i + len("window.TASKS = ") :].rstrip().rstrip(";"))
    return tasks, ds


def used_view_keys(tasks: list) -> set:
    return {str(t.get("view_key") or "") for t in tasks}


def discover_new_snapshots(used: set) -> list[tuple[Path, LayoutState, str]]:
    out = []
    seen = set(used)
    for p in sorted((ROOT / "data" / "datasets").glob("*/snapshots/*.state.json")):
        if p.parent.parent.name.startswith("_tmp") or p.parent.parent.name == "demo":
            continue
        try:
            st = LayoutState.model_validate(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
        vk = view_key(st)
        if vk in seen:
            continue
        if not _official_drawing_url(st):
            continue
        seen.add(vk)
        out.append((p, st, vk))
    return out


def append_t1(new: list[tuple[Path, LayoutState, str]]) -> int:
    pack = ROOT / "data" / "evalsys" / "annotate_t1"
    tasks, dsid = load_tasks(pack)
    rend = pack / "renders"
    rend.mkdir(parents=True, exist_ok=True)
    n = 0
    for p, st, vk in new:
        sid = p.stem
        dst = rend / f"{sid}.png"
        try:
            render_preview(st, dst)
        except Exception as e:  # noqa: BLE001
            print(f"t1 render fail {p.name}: {e}")
            continue
        task = {
            "id": sid,
            "type": "grade",
            "view_key": vk,
            "drawing_url": _official_drawing_url(st),
            "tenant_id": str((st.source or {}).get("tenant_id") or ""),
            "render": f"renders/{sid}.png",
            "state": str(p.resolve().as_posix()),
        }
        _enrich_task(task, vk)
        if not (task.get("part_name") or task.get("tree_uuid")):
            print(f"t1 skip {p.name}: no online part")
            if dst.exists():
                dst.unlink()
            continue
        tasks.append(task)
        n += 1
    shutil.copy(ROOT / "tools" / "annotate.html", pack / "annotate.html")
    _write_package(pack, tasks, dsid)
    print(f"annotate_t1: added {n}, total {len(tasks)}")
    return n


def append_t2_from_manifest() -> int:
    pack = ROOT / "data" / "evalsys" / "annotate_t2"
    tasks, dsid = load_tasks(pack)
    have = {t["id"] for t in tasks}
    used = used_view_keys(tasks)
    man = ROOT / "data" / "evalsys" / "pairs_perturb" / "manifest.jsonl"
    if not man.exists():
        print("no pairs manifest")
        return 0
    rend = pack / "renders"
    rend.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        pid = rec.get("pair_id")
        vk = str(rec.get("view_key") or "")
        if not pid or pid in have:
            continue
        if not LIVE.match(vk):
            continue
        if vk in used and pid not in have:
            # same view already in T2 from an older pair — still ok to add another pair
            pass
        sa = ROOT / "data" / "evalsys" / "pairs_perturb" / rec["state_a"]
        sb = ROOT / "data" / "evalsys" / "pairs_perturb" / rec["state_b"]
        if not sa.exists() or not sb.exists():
            continue
        # only append pairs whose view is new (not already labeled in this pack)
        if vk in used:
            continue
        ra_dst = rend / f"{pid}_L.png"
        rb_dst = rend / f"{pid}_R.png"
        # Always render with the current preview.py. Copying pairs_perturb PNGs
        # reintroduces stale styles (black/yellow, text sitting on the dim line).
        st_a = LayoutState.model_validate(json.loads(sa.read_text(encoding="utf-8")))
        st_b = LayoutState.model_validate(json.loads(sb.read_text(encoding="utf-8")))
        try:
            render_preview(st_a, ra_dst)
            render_preview(st_b, rb_dst)
        except Exception as e:  # noqa: BLE001
            print(f"t2 render fail {pid}: {e}")
            continue
        st = st_a
        task = {
            "id": pid,
            "type": "prefer",
            "view_key": vk,
            "drawing_url": _official_drawing_url(st),
            "tenant_id": str((st.source or {}).get("tenant_id") or ""),
            "render_a": f"renders/{pid}_L.png",
            "render_b": f"renders/{pid}_R.png",
            "state_a": str(sa.resolve().as_posix()),
            "state_b": str(sb.resolve().as_posix()),
        }
        _enrich_task(task, vk)
        tasks.append(task)
        used.add(vk)
        have.add(pid)
        n += 1
    shutil.copy(ROOT / "tools" / "annotate.html", pack / "annotate.html")
    _write_package(pack, tasks, dsid)
    print(f"annotate_t2: added {n}, total {len(tasks)}")
    return n


def rerender_pack(pack: Path) -> int:
    """Redraw every task PNG from its LayoutState using the current preview."""
    tasks, _ = load_tasks(pack)
    rend = pack / "renders"
    rend.mkdir(parents=True, exist_ok=True)
    n = 0
    fails = 0
    total = sum(2 if t.get("type") == "prefer" else 1 for t in tasks)
    for t in tasks:
        if t.get("type") == "prefer":
            pairs = ((t.get("state_a"), t.get("render_a")), (t.get("state_b"), t.get("render_b")))
        else:
            pairs = ((t.get("state"), t.get("render")),)
        for sp, rp in pairs:
            if not sp or not rp:
                continue
            src = Path(sp)
            dst = rend / Path(rp).name
            if not src.exists():
                print(f"missing state {src}")
                fails += 1
                continue
            try:
                st = LayoutState.model_validate(json.loads(src.read_text(encoding="utf-8")))
                render_preview(st, dst)
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"render fail {dst.name}: {e}")
                fails += 1
            if n % 20 == 0:
                print(f"  {pack.name}: {n}/{total}")
    print(f"re-rendered {n} images in {pack.name} ({fails} failed)")
    return n


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rerender",
        nargs="*",
        metavar="PACK",
        help="Redraw PNGs in annotate_t1 / annotate_t2 (default: t2). Does not reshuffle task ids.",
    )
    args, _ = ap.parse_known_args()
    if args.rerender is not None:
        names = args.rerender or ["t2"]
        for name in names:
            pack = name if name.startswith("annotate_") else f"annotate_{name}"
            rerender_pack(ROOT / "data" / "evalsys" / pack)
        return
    t1, _ = load_tasks(ROOT / "data" / "evalsys" / "annotate_t1")
    t2, _ = load_tasks(ROOT / "data" / "evalsys" / "annotate_t2")
    used = used_view_keys(t1) | used_view_keys(t2)
    new = discover_new_snapshots(used)
    print(f"new live snapshots: {len(new)}")
    append_t1(new)
    append_t2_from_manifest()


if __name__ == "__main__":
    main()
