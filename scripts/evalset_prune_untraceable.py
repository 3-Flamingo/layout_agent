#!/usr/bin/env python3
"""Drop annotation tasks that cannot open a live 官网投图 (SYN / oss / no solutionId)."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evalset_make_annotation_tasks import _official_drawing_url, _enrich_task  # noqa: E402
from layout_agent.models import LayoutState  # noqa: E402

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


def state_paths(task: dict) -> list[Path]:
    out = []
    for k in ("state", "state_a", "state_b"):
        p = task.get(k)
        if p:
            out.append(Path(p))
    return out


def keep_task(task: dict) -> bool:
    vk = str(task.get("view_key") or "")
    if LIVE.match(vk):
        return True
    url = str(task.get("drawing_url") or "")
    return url.startswith("http")


def is_t1_synthetic(task: dict) -> bool:
    """Perturbed synthetic states: 官网投图 is the original, not this image."""
    st = str(task.get("state") or "").replace("\\", "/")
    if "pairs_perturb" in st:
        return True
    ident = str(task.get("id") or "")
    return bool(re.search(r"_(light|medium|heavy)\d+_b\.state$", ident))


def has_online_part(task: dict) -> bool:
    return bool(str(task.get("part_name") or "").strip() or str(task.get("tree_uuid") or "").strip())


def backfill(task: dict) -> None:
    for p in state_paths(task):
        if not p.exists():
            continue
        st = LayoutState.model_validate(json.loads(p.read_text(encoding="utf-8")))
        url = _official_drawing_url(st)
        if url:
            task["drawing_url"] = url
        src = st.source or {}
        if src.get("tenant_id"):
            task["tenant_id"] = str(src["tenant_id"])
        _enrich_task(task, task.get("view_key") or "")
        break


def prune_pack(pack: Path) -> None:
    tasks, dsid = load_tasks(pack)
    kept = []
    dropped = []
    for t in tasks:
        if pack.name == "annotate_t1" and is_t1_synthetic(t):
            dropped.append(t["id"])
            continue
        if keep_task(t):
            backfill(t)
            if pack.name == "annotate_t1" and not has_online_part(t):
                dropped.append(t["id"])
                continue
            kept.append(t)
        else:
            dropped.append(t["id"])
    shutil.copy(ROOT / "tools" / "annotate.html", pack / "annotate.html")
    js = f"window.DATASET_ID = {json.dumps(dsid)};\nwindow.TASKS = {json.dumps(kept, ensure_ascii=False)};\n"
    (pack / "tasks.js").write_text(js, encoding="utf-8")
    rend = pack / "renders"
    used = set()
    for t in kept:
        for k in ("render", "render_a", "render_b"):
            rel = t.get(k)
            if rel:
                used.add(Path(rel).name)
    n_del = 0
    if rend.is_dir():
        for png in rend.glob("*.png"):
            if png.name not in used:
                png.unlink()
                n_del += 1
    print(f"{pack.name}: {len(tasks)} -> {len(kept)} kept, dropped {len(dropped)}, removed {n_del} pngs")
    if dropped:
        print("  dropped:", ", ".join(dropped[:12]) + (" ..." if len(dropped) > 12 else ""))


def prune_holdout() -> None:
    p = ROOT / "data" / "evalsys" / "holdout_views.txt"
    if not p.exists():
        return
    lines = p.read_text(encoding="utf-8").splitlines()
    keep = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            keep.append(ln)
            continue
        if LIVE.match(s):
            keep.append(ln)
        else:
            print(f"holdout drop {s}")
    p.write_text("\n".join(keep).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    for name in ("annotate_t1", "annotate_t2"):
        prune_pack(ROOT / "data" / "evalsys" / name)
    prune_holdout()


if __name__ == "__main__":
    main()
