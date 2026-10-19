#!/usr/bin/env python3
"""Import colleague-arranged 中车 pairs (perturbed | perfect) into T1/T2.

Source:
  https://v3.private.designorder.cn/drawing/?solutionId=1537422555585998848&tenantId=250

Sheets are laid out left-to-right, top-to-bottom. Consecutive drawings are
(perturbed, perfect). Matching CadViews share a uuid; numeric element ids differ.
Screenshots use CadView toPicture (no Python preview, no extra perturbation).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_production_dataset import (  # noqa: E402
    _eid,
    _mat_tx,
    build_index,
    convert_view,
    load_json_file,
)
from layout_agent.perceive import ACTIVE_PALETTE  # noqa: E402
from evalset_make_annotation_tasks import _enrich_task, _official_drawing_url, _write_package  # noqa: E402
from evalset_append_new_views import load_tasks  # noqa: E402
from layout_agent.models import LayoutState  # noqa: E402

SID = "1537422555585998848"
TENANT = "250"
DRAWING_HOST = "https://v3.private.designorder.cn/drawing"
ROW_TOL = 200.0
DS_DIR = ROOT / "data" / "datasets" / "prod_colleague_pairs_598848"
RAW = ROOT / "data" / "raw" / f"solution_{SID}"
OFFICIAL = ROOT / "data" / "evalsys" / "official_renders"


def _safe(uuid: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in uuid)


def drawing_order(by: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for d in by.get("CadDrawing", []):
        tx, ty = _mat_tx(d.get("matrix"))
        rows.append({"eid": _eid(d.get("id")), "tx": tx, "ty": ty})
    ordered = sorted(rows, key=lambda r: (-r["ty"], r["tx"]))
    clusters: List[List[Dict[str, Any]]] = []
    cur = [ordered[0]]
    for r in ordered[1:]:
        if abs(r["ty"] - cur[0]["ty"]) <= ROW_TOL:
            cur.append(r)
        else:
            clusters.append(sorted(cur, key=lambda x: x["tx"]))
            cur = [r]
    clusters.append(sorted(cur, key=lambda x: x["tx"]))
    flat: List[Dict[str, Any]] = []
    for i, cl in enumerate(clusters):
        if len(cl) % 2:
            print(f"WARN drawing row {i} has odd count {len(cl)}")
        for j, d in enumerate(cl):
            d["row"] = i
            d["col"] = j
            flat.append(d)
    return flat


def view_dim_count(vid: int, by: Dict[str, List[Dict[str, Any]]]) -> int:
    n = 0
    for ctor in ("CadAlignedDimension", "CadDiameterDimension", "CadRadialDimension"):
        n += sum(1 for el in by.get(ctor, []) if el.get("pid") == vid and not el.get("inVisibleFlag"))
    return n


def views_on_drawing(
    drawing_eid: int,
    by: Dict[str, List[Dict[str, Any]]],
    groups_by_drawing: Dict[int, set],
) -> Dict[str, Dict[str, Any]]:
    vg_ids = groups_by_drawing.get(drawing_eid, set())
    out: Dict[str, Dict[str, Any]] = {}
    for v in by.get("CadView", []):
        if _eid(v.get("pid")) not in vg_ids:
            continue
        if v.get("inVisibleFlag"):
            continue
        vid = _eid(v.get("id"))
        n = view_dim_count(vid, by)
        if n <= 0 or vid is None:
            continue
        out[str(v.get("uuid") or "")] = {"eid": vid, "view": v, "dims": n}
    out.pop("", None)
    return out


def build_pairs(by: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    groups_by_drawing: Dict[int, set] = defaultdict(set)
    for g in by.get("CadViewGroup", []):
        de, ge = _eid(g.get("pid")), _eid(g.get("id"))
        if de is not None and ge is not None:
            groups_by_drawing[de].add(ge)
    order = drawing_order(by)
    pairs: List[Dict[str, Any]] = []
    for i in range(0, len(order) - 1, 2):
        left, right = order[i], order[i + 1]
        lv = views_on_drawing(left["eid"], by, groups_by_drawing)
        rv = views_on_drawing(right["eid"], by, groups_by_drawing)
        for u in sorted(set(lv) & set(rv)):
            pairs.append(
                {
                    "pair_index": i // 2,
                    "uuid": u,
                    "viewType": lv[u]["view"].get("viewType"),
                    "perturbed_eid": lv[u]["eid"],
                    "perfect_eid": rv[u]["eid"],
                    "left_drawing": left["eid"],
                    "right_drawing": right["eid"],
                }
            )
    return pairs


def convert_all(pairs: List[Dict[str, Any]], by: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    meta = json.loads((RAW / "solution_meta.json").read_text(encoding="utf-8"))
    meta["_tenant_id"] = TENANT
    meta["_drawing_host"] = DRAWING_HOST
    palette = ACTIVE_PALETTE * 20
    by_eid = {_eid(v.get("id")): v for v in by.get("CadView", [])}
    snap = DS_DIR / "snapshots"
    snap.mkdir(parents=True, exist_ok=True)
    out_pairs: List[Dict[str, Any]] = []
    for rec in pairs:
        item = dict(rec)
        ok = True
        for role, eid_key in (("perturbed", "perturbed_eid"), ("perfect", "perfect_eid")):
            eid = rec[eid_key]
            view = by_eid.get(eid)
            if view is None:
                print(f"missing view eid={eid}")
                ok = False
                break
            try:
                st = convert_view(view, by, meta, palette)
            except Exception as e:  # noqa: BLE001
                print(f"convert fail eid={eid} uuid={rec['uuid']}: {e}")
                ok = False
                break
            if st is None:
                ok = False
                break
            safe = _safe(str(rec["uuid"]))
            st.state_id = f"view_{safe}_{eid}"
            st.source = dict(st.source or {})
            st.source["cad_view_id"] = str(eid)
            st.source["colleague_role"] = role
            st.source["drawing_url"] = f"{DRAWING_HOST}/?solutionId={SID}&tenantId={TENANT}"
            rel = f"snapshots/{st.state_id}.state.json"
            (DS_DIR / rel).write_text(
                json.dumps(st.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            item[f"{role}_state"] = rel
            item[f"{role}_state_id"] = st.state_id
        if ok:
            item["pair_id"] = f"colleague_{_safe(str(rec['uuid']))}_{rec['pair_index']}"
            out_pairs.append(item)
    man = {
        "dataset_name": "prod_colleague_pairs_598848",
        "solution_id": SID,
        "tenant_id": TENANT,
        "drawing_url": f"{DRAWING_HOST}/?solutionId={SID}&tenantId={TENANT}",
        "note": "Colleague-arranged perturbed|perfect pairs on one 中车 sheet. Left=perturbed, right=perfect.",
        "n_pairs": len(out_pairs),
        "pairs": out_pairs,
    }
    DS_DIR.mkdir(parents=True, exist_ok=True)
    (DS_DIR / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"converted {len(out_pairs)} pairs -> {DS_DIR}")
    return out_pairs


def _b64_png(data_url: str) -> bytes:
    import re
    import base64

    m = re.match(r"data:image/[^;]+;base64,(.+)$", data_url, re.S)
    if not m:
        raise ValueError("expected data:image/...;base64")
    return base64.b64decode(m.group(1))


def capture_official(pairs: List[Dict[str, Any]], origin: str, overwrite: bool) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise SystemExit("pip install playwright && python -m playwright install chromium") from e

    jobs: List[Tuple[int, Path]] = []
    for rec in pairs:
        for role in ("perturbed", "perfect"):
            sid = rec[f"{role}_state_id"]
            eid = rec[f"{role}_eid"]
            dst = OFFICIAL / f"{sid}.png"
            if dst.exists() and not overwrite:
                continue
            jobs.append((eid, dst))
    if not jobs:
        print("all official screenshots already present")
        return 0
    url = f"{origin.rstrip('/')}/?solutionId={SID}&tenantId={TENANT}&layoutEvalSandbox=1"
    print(f"open {url}  ({len(jobs)} shots)")
    n = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context(viewport={"width": 1600, "height": 1000}).new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=300000)
        page.wait_for_function("() => window.__layoutEval", timeout=120000)
        page.wait_for_function(
            "() => window.__layoutEval.listViews().length > 0",
            timeout=300000,
        )
        nviews = page.evaluate("() => window.__layoutEval.listViews().length")
        print(f"  views_on_sheet={nviews}")
        page.evaluate("() => window.__layoutEval.prepareUi()")
        page.wait_for_timeout(8000)
        for eid, dst in jobs:
            print(f"  capture eid={eid} -> {dst.name}", flush=True)
            res = page.evaluate(
                """async (spec) => window.__layoutEval.captureView(spec)""",
                {"view_id": eid},
            )
            png = res.get("png") if isinstance(res, dict) else None
            if (not png) and isinstance(res, dict) and res.get("ok"):
                page.wait_for_timeout(1500)
                res = page.evaluate(
                    """async (spec) => window.__layoutEval.captureView(spec)""",
                    {"view_id": eid},
                )
                png = res.get("png") if isinstance(res, dict) else None
            if not res or not png:
                page.evaluate("() => window.__layoutEval.prepareUi()")
                page.wait_for_timeout(300)
                dst.parent.mkdir(parents=True, exist_ok=True)
                vp = page.locator("[data-layout-eval-viewport] canvas")
                if vp.count() == 0:
                    vp = page.locator("#view3d canvas")
                if vp.count() == 0:
                    err = {k: res.get(k) for k in ("ok", "error")} if isinstance(res, dict) else res
                    print(f"    FAIL {eid}: {err}")
                    continue
                vp.first.screenshot(path=str(dst), type="png")
                extra = " via=canvas"
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(_b64_png(png))
                extra = " via=paper"
            n += 1
            print(f"    wrote {dst.name}{extra}")
        browser.close()
    print(f"captured {n}/{len(jobs)}")
    return n


def append_packages(pairs: List[Dict[str, Any]], seed: int) -> None:
    t2, ds2 = load_tasks(ROOT / "data" / "evalsys" / "annotate_t2")
    t1, ds1 = load_tasks(ROOT / "data" / "evalsys" / "annotate_t1")
    have2 = {t["id"] for t in t2}
    have1 = {t["id"] for t in t1}
    rend2 = ROOT / "data" / "evalsys" / "annotate_t2" / "renders"
    rend1 = ROOT / "data" / "evalsys" / "annotate_t1" / "renders"
    rend2.mkdir(parents=True, exist_ok=True)
    rend1.mkdir(parents=True, exist_ok=True)
    n2 = n1 = 0
    missing = 0
    for rec in pairs:
        pid = rec["pair_id"]
        vk = f"{SID}/{rec['uuid']}"
        sp_p = DS_DIR / rec["perturbed_state"]
        sp_k = DS_DIR / rec["perfect_state"]
        png_p = OFFICIAL / f"{rec['perturbed_state_id']}.png"
        png_k = OFFICIAL / f"{rec['perfect_state_id']}.png"
        if not png_p.exists() or not png_k.exists():
            missing += 1
            print(f"skip {pid}: missing official png")
            continue
        st = LayoutState.model_validate(json.loads(sp_p.read_text(encoding="utf-8")))
        url = _official_drawing_url(st)
        if pid not in have2:
            shutil.copy(png_p, rend2 / f"{pid}_L.png")
            shutil.copy(png_k, rend2 / f"{pid}_R.png")
            task = {
                "id": pid,
                "type": "prefer",
                "view_key": vk,
                "drawing_url": url,
                "tenant_id": TENANT,
                "render_a": f"renders/{pid}_L.png",
                "render_b": f"renders/{pid}_R.png",
                "state_a": str(sp_p.resolve().as_posix()),
                "state_b": str(sp_k.resolve().as_posix()),
                "source": "colleague_pairs",
                "colleague_role_a": "perturbed",
                "colleague_role_b": "perfect",
                "gold_prefer": "b",
            }
            _enrich_task(task, vk)
            t2.append(task)
            have2.add(pid)
            n2 += 1
        for role, sp, png in (
            ("perturbed", sp_p, png_p),
            ("perfect", sp_k, png_k),
        ):
            tid = Path(rec[f"{role}_state"]).stem  # view_xxx.state
            if tid in have1:
                continue
            dst = rend1 / f"{tid}.png"
            shutil.copy(png, dst)
            task = {
                "id": tid,
                "type": "grade",
                "view_key": vk,
                "drawing_url": url,
                "tenant_id": TENANT,
                "render": f"renders/{tid}.png",
                "state": str(sp.resolve().as_posix()),
                "source": "colleague_pairs",
                "colleague_role": role,
            }
            _enrich_task(task, vk)
            t1.append(task)
            have1.add(tid)
            n1 += 1
    shutil.copy(ROOT / "tools" / "annotate.html", ROOT / "data" / "evalsys" / "annotate_t2" / "annotate.html")
    shutil.copy(ROOT / "tools" / "annotate.html", ROOT / "data" / "evalsys" / "annotate_t1" / "annotate.html")
    _write_package(ROOT / "data" / "evalsys" / "annotate_t2", t2, ds2)
    _write_package(ROOT / "data" / "evalsys" / "annotate_t1", t1, ds1)
    print(f"annotate_t2 +{n2} total {len(t2)}; annotate_t1 +{n1} total {len(t1)}; missing_png {missing}")


def _role_of_state(path: Path) -> str:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return str((raw.get("source") or {}).get("colleague_role") or "")


def align_t2_ab(pairs: List[Dict[str, Any]]) -> None:
    """Force colleague T2 items to A=perturbed, B=perfect. Record ids that were flipped."""
    pack = ROOT / "data" / "evalsys" / "annotate_t2"
    tasks, dsid = load_tasks(pack)
    by_id = {t["id"]: t for t in tasks}
    rend = pack / "renders"
    flip_ids: List[str] = []
    rows: List[Dict[str, Any]] = []
    n = 0
    for rec in pairs:
        pid = rec["pair_id"]
        t = by_id.get(pid)
        if not t:
            continue
        role_a = _role_of_state(Path(t["state_a"]))
        role_b = _role_of_state(Path(t["state_b"]))
        if role_a == "perfect" and role_b == "perturbed":
            flip_ids.append(pid)
        png_p = OFFICIAL / f"{rec['perturbed_state_id']}.png"
        png_k = OFFICIAL / f"{rec['perfect_state_id']}.png"
        if not png_p.exists() or not png_k.exists():
            print(f"skip align {pid}: missing png")
            continue
        shutil.copy(png_p, rend / f"{pid}_L.png")
        shutil.copy(png_k, rend / f"{pid}_R.png")
        t["state_a"] = str((DS_DIR / rec["perturbed_state"]).resolve().as_posix())
        t["state_b"] = str((DS_DIR / rec["perfect_state"]).resolve().as_posix())
        t["source"] = "colleague_pairs"
        t["colleague_role_a"] = "perturbed"
        t["colleague_role_b"] = "perfect"
        t["gold_prefer"] = "b"
        rows.append(
            {
                "pair_id": pid,
                "before_A": role_a,
                "before_B": role_b,
                "after_A": "perturbed",
                "after_B": "perfect",
                "label_flip": pid in flip_ids,
                "gold_prefer": "b",
            }
        )
        n += 1
    map_path = ROOT / "data" / "evalsys" / "colleague_t2_ab_map.json"
    map_path.write_text(
        json.dumps(
            {
                "note": "Colleague T2 pairs aligned to A=perturbed, B=perfect. gold_prefer=b. "
                "label_flip=true means the pre-align display had A=perfect; browser labels a/b "
                "for those ids are remapped once on next page load.",
                "n": n,
                "n_flipped": len(flip_ids),
                "flip_ids": flip_ids,
                "items": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.copy(ROOT / "tools" / "annotate.html", pack / "annotate.html")
    js = (
        f"window.DATASET_ID = {json.dumps(dsid)};\n"
        f"window.COLLEAGUE_LABEL_FLIP = {json.dumps(flip_ids)};\n"
        f"window.TASKS = {json.dumps(tasks, ensure_ascii=False)};\n"
    )
    (pack / "tasks.js").write_text(js, encoding="utf-8")
    print(f"aligned {n} colleague T2 pairs; flipped {len(flip_ids)}; map -> {map_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--align-ab", action="store_true", help="reorder colleague T2 to A=perturbed, B=perfect")
    ap.add_argument("--origin", default="http://127.0.0.1:8080")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    if not (args.convert or args.capture or args.append or args.align_ab):
        args.convert = args.capture = args.append = True

    if args.convert or args.capture or args.append:
        doc = load_json_file(RAW / "solution_doc.json")
        by = build_index(doc)
        if args.convert or not (DS_DIR / "manifest.json").exists():
            pairs_meta = build_pairs(by)
            print(f"sheet-derived view pairs: {len(pairs_meta)}")
            pairs = convert_all(pairs_meta, by)
        else:
            man = json.loads((DS_DIR / "manifest.json").read_text(encoding="utf-8"))
            pairs = man.get("pairs") or []
            print(f"reuse converted pairs: {len(pairs)}")
        if args.capture:
            capture_official(pairs, args.origin, args.overwrite)
        if args.append:
            append_packages(pairs, args.seed)
    else:
        man = json.loads((DS_DIR / "manifest.json").read_text(encoding="utf-8"))
        pairs = man.get("pairs") or []

    if args.align_ab:
        if "pairs" not in dir():
            man = json.loads((DS_DIR / "manifest.json").read_text(encoding="utf-8"))
            pairs = man.get("pairs") or []
        align_t2_ab(pairs)


if __name__ == "__main__":
    main()
