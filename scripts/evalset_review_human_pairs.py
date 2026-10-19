#!/usr/bin/env python3
"""Build a local HTML package to inspect human_pairs train/test splits.

Open:
  data/evalsys/review_human_pairs/review.html

Left = defective layout, right = perfect. No left/right randomization.
Shows the frozen test set, train (real + in-train CMU synth), and cross-view holdout.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.evalset.human_gold import read_jsonl
from layout_agent.evalset.preview import render_preview
from layout_agent.models import LayoutState

sys.path.insert(0, str(ROOT / "scripts"))
from evalset_make_annotation_tasks import load_view_tree_lookup  # noqa: E402

HP = ROOT / "data" / "evalsys" / "human_pairs"
OUT = ROOT / "data" / "evalsys" / "review_human_pairs"
HTML_SRC = ROOT / "tools" / "review_human_pairs.html"


def _load_state(rel: str) -> LayoutState:
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / rel
    return LayoutState.model_validate(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-px", type=int, default=1100)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    rend = out / "renders"
    rend.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    test_path = HP / "test.jsonl"
    train_path = HP / "train_with_synth.jsonl"
    if not train_path.exists():
        train_path = HP / "train.jsonl"
    cross_path = HP / "cross_view.jsonl"
    if test_path.exists():
        rows.extend(read_jsonl(test_path))
    if train_path.exists():
        rows.extend(read_jsonl(train_path))
    if cross_path.exists():
        rows.extend(read_jsonl(cross_path))

    tree = load_view_tree_lookup()
    tasks: List[Dict[str, Any]] = []
    for rec in rows:
        pid = str(rec["pair_id"])
        dst_a = rend / f"{pid}_a.png"
        dst_b = rend / f"{pid}_b.png"
        st_a = _load_state(str(rec["state_a"]))
        st_b = _load_state(str(rec["state_b"]))
        if not (args.skip_existing and dst_a.exists()):
            render_preview(st_a, dst_a, max_px=args.max_px)
        if not (args.skip_existing and dst_b.exists()):
            render_preview(st_b, dst_b, max_px=args.max_px)
        src = st_a.source or {}
        sid = str(rec.get("solution_id") or src.get("solution_id") or "")
        tenant = str(rec.get("tenant_id") or src.get("tenant_id") or "")
        vkey = str(rec.get("view_key") or "")
        vu = vkey.split("/")[-1]
        tree_meta = tree.get(vkey) or {}
        part_name = (
            tree_meta.get("part_name")
            or src.get("part_number")
            or rec.get("part_name")
            or ""
        )
        tree_uuid = tree_meta.get("tree_uuid") or rec.get("tree_uuid") or ""
        if tenant == "250":
            drawing_url = (
                f"https://v3.private.designorder.cn/drawing/?solutionId={sid}&tenantId=250"
            )
            project_url = ""
        else:
            drawing_url = (
                f"https://v3.designorder.cn/drawing/?solutionId={sid}&tenantId={tenant or '3'}"
            )
            project_url = (
                "https://v3.designorder.cn/shanshe-enterprise/project/"
                "2099378950165704704?tenantId=3&app=shanshe-enterprise"
            )
        drawing_url_alt = (
            f"https://drawing.designorder.cn/?solutionId={sid}&tenantId={tenant or '3'}"
            if tenant != "250"
            else ""
        )
        tasks.append(
            {
                "id": pid,
                "type": "prefer",
                "split": rec.get("split"),
                "source": rec.get("source"),
                "view_key": rec.get("view_key"),
                "view_uuid": vu,
                "part_name": part_name,
                "tree_uuid": tree_uuid,
                "solution_id": sid,
                "solution_name": rec.get("solution_name") or src.get("solution_name") or src.get("drawing_id") or "",
                "tenant_id": tenant,
                "view_type": rec.get("view_type") or src.get("view_type"),
                "n_ann": rec.get("n_ann"),
                "pose_delta": rec.get("pose_delta"),
                "quality": rec.get("quality"),
                "drawing_url": drawing_url,
                "drawing_url_alt": drawing_url_alt,
                "project_url": project_url,
                "perturbed_eid": rec.get("perturbed_eid") or src.get("cad_view_id"),
                "perfect_eid": rec.get("perfect_eid"),
                "sides_swapped": bool(rec.get("sides_swapped")),
                "render_a": f"renders/{pid}_a.png",
                "render_b": f"renders/{pid}_b.png",
                "state_a": rec.get("state_a"),
                "state_b": rec.get("state_b"),
                "caption_a": "左 · 缺陷",
                "caption_b": "右 · 完美",
                "preflag": rec.get("review_flag") or "",
            }
        )
        print(f"rendered {pid} split={rec.get('split')} delta={rec.get('pose_delta')}")

    shutil.copy(HTML_SRC, out / "review.html")
    n_test = sum(1 for t in tasks if t.get("split") == "test")
    n_train = sum(1 for t in tasks if t.get("split") == "train")
    n_train_real = sum(1 for t in tasks if t.get("split") == "train" and t.get("source") != "cmu_perturb_perfect")
    n_train_synth = sum(1 for t in tasks if t.get("split") == "train" and t.get("source") == "cmu_perturb_perfect")
    n_cross = sum(1 for t in tasks if t.get("split") == "cross_view")
    summary = {
        "test": n_test,
        "train": n_train,
        "train_real": n_train_real,
        "train_synth": n_train_synth,
        "cross_view": n_cross,
        "all": len(tasks),
    }
    js = (
        "window.DATASET_ID = \"review_human_pairs\";\n"
        f"window.SUMMARY = {json.dumps(summary, ensure_ascii=False)};\n"
        f"window.TASKS = {json.dumps(tasks, ensure_ascii=False)};\n"
    )
    (out / "tasks.js").write_text(js, encoding="utf-8")
    print(
        f"open {out / 'review.html'}  test={n_test} train={n_train} "
        f"(real={n_train_real} synth={n_train_synth}) cross_view={n_cross}"
    )


if __name__ == "__main__":
    main()
