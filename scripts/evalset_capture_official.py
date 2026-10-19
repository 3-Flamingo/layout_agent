#!/usr/bin/env python3
"""Capture T1/T2 images from the local official drawing frontend (CadView 投图).

Perturbed layouts do not exist on the production website. This script loads the
same frontend as the official 投图, writes LayoutState poses back onto CadView,
then takes toPicture screenshots.

Prereq:
  1. yarn start:drawing  (http://127.0.0.1:8080, env.js matches the solution)
  2. pip install playwright && python -m playwright install chromium

Usage:
  python scripts/evalset_capture_official.py --from-t2 --from-t1 --limit 1
  python scripts/evalset_capture_official.py --from-t2 --from-t1
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _b64_png(data_url: str) -> bytes:
    m = re.match(r"data:image/[^;]+;base64,(.+)$", data_url, re.S)
    if not m:
        raise ValueError("expected data:image/...;base64")
    return base64.b64decode(m.group(1))


def _tenant(src: Dict[str, Any]) -> str:
    t = src.get("tenant_id")
    if t not in (None, ""):
        return str(t)
    url = str(src.get("drawing_url") or "")
    m = re.search(r"tenantId=(\d+)", url)
    if m:
        return m.group(1)
    if "private.designorder.cn" in url:
        return "250"
    return "0"


def _live_solution(src: Dict[str, Any]) -> Optional[str]:
    sid = str(src.get("solution_id") or "")
    if re.fullmatch(r"\d{10,}", sid):
        return sid
    return None


def _jobs_from_tasks_js(path: Path) -> List[Path]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    m = re.search(r"window\.TASKS\s*=\s*(\[.*\]);", text, re.S)
    if not m:
        return []
    tasks = json.loads(m.group(1))
    out: List[Path] = []
    for t in tasks:
        for key in ("state", "state_a", "state_b"):
            p = t.get(key)
            if p:
                out.append(Path(p))
    return out


def collect_state_files(args: argparse.Namespace) -> List[Path]:
    files: List[Path] = []
    if args.from_t2:
        files.extend(_jobs_from_tasks_js(ROOT / "data" / "evalsys" / "annotate_t2" / "tasks.js"))
    if args.from_t1:
        files.extend(_jobs_from_tasks_js(ROOT / "data" / "evalsys" / "annotate_t1" / "tasks.js"))
    import glob as _glob

    for g in args.states or []:
        files.extend(Path(p) for p in _glob.glob(g, recursive=True))
    uniq: List[Path] = []
    seen = set()
    for p in files:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            uniq.append(p)
    return uniq


def group_jobs(
    states: List[Path],
    out_dir: Path,
) -> Dict[Tuple[str, str], List[Tuple[Path, Path, Dict[str, Any]]]]:
    groups: Dict[Tuple[str, str], List[Tuple[Path, Path, Dict[str, Any]]]] = defaultdict(list)
    skipped = 0
    for p in states:
        raw = json.loads(p.read_text(encoding="utf-8"))
        src = raw.get("source") or {}
        sid = _live_solution(src)
        if not sid:
            skipped += 1
            print(f"skip synthetic/non-live {p.name}")
            continue
        tenant = _tenant(src)
        dst = out_dir / f"{p.name[: -len('.state.json')] if p.name.endswith('.state.json') else p.stem}.png"
        groups[(sid, tenant)].append((p, dst, raw))
    if skipped:
        print(f"skipped {skipped} states without a live solutionId")
    return groups


def capture_group(page, origin: str, sid: str, tenant: str, jobs: List[Tuple[Path, Path, Dict[str, Any]]], skip_existing: bool) -> None:
    pending = [(sp, dst, raw) for sp, dst, raw in jobs if not (skip_existing and dst.exists())]
    if not pending:
        print(f"  {sid} all {len(jobs)} already captured")
        return
    url = (
        f"{origin.rstrip('/')}/?solutionId={sid}&tenantId={tenant}"
        f"&layoutEvalSandbox=1"
    )
    print(f"open {url}  ({len(pending)} shots)")
    page.goto(url, wait_until="domcontentloaded", timeout=180000)
    page.wait_for_function("() => window.__layoutEval", timeout=180000)
    ready = page.evaluate("() => window.__layoutEval.waitReady(180000)")
    if not ready or not ready.get("ok"):
        raise RuntimeError(f"drawing not ready for {sid}: {ready}")
    page.evaluate("() => window.__layoutEval.prepareUi()")
    page.wait_for_timeout(400)
    print(f"  views={len(ready.get('views') or [])}")
    for sp, dst, raw in pending:
        print(f"  capture {sp.name} ...", flush=True)
        res = page.evaluate(
            """async (state) => window.__layoutEval.applyAndZoom(state)""",
            raw,
        )
        if not res or not res.get("ok"):
            raise RuntimeError(f"{sp.name}: {res}")
        page.evaluate("() => window.__layoutEval.prepareUi()")
        page.wait_for_timeout(200)
        dst.parent.mkdir(parents=True, exist_ok=True)
        png_url = res.get("png") if isinstance(res, dict) else None
        if png_url:
            dst.write_bytes(_b64_png(png_url))
            extra = f" applied={len(res.get('applied') or [])} via=paper"
        else:
            vp = page.locator("[data-layout-eval-viewport] canvas")
            if vp.count() == 0:
                vp = page.locator("#view3d canvas")
            vp.first.screenshot(path=str(dst), type="png")
            extra = f" applied={len(res.get('applied') or [])} via=canvas"
        if res.get("error"):
            extra += f" warn={res['error']}"
        print(f"    wrote {dst.name}{extra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default=str(ROOT / "data" / "evalsys" / "official_renders"))
    ap.add_argument("--from-t2", action="store_true")
    ap.add_argument("--from-t1", action="store_true")
    ap.add_argument("--states", action="append", default=[], help="glob of *.state.json")
    ap.add_argument("--only-solution", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise SystemExit("pip install playwright && python -m playwright install chromium") from e

    states = collect_state_files(args)
    if not states:
        raise SystemExit("no state files (pass --from-t2 / --from-t1 / --states)")
    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    groups = group_jobs(states, out_dir)
    if args.only_solution:
        groups = {k: v for k, v in groups.items() if k[0] == args.only_solution}
    if args.limit:
        kept = {}
        n = 0
        for k, jobs in groups.items():
            if n >= args.limit:
                break
            take = jobs[: max(0, args.limit - n)]
            if take:
                kept[k] = take
                n += len(take)
        groups = kept
    total = sum(len(v) for v in groups.values())
    print(f"solutions={len(groups)} shots={total} out={out_dir}")

    origin = args.origin
    host = urlparse(origin).hostname
    if host not in ("127.0.0.1", "localhost"):
        raise SystemExit("origin must be local drawing (127.0.0.1 / localhost)")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()
        try:
            for (sid, tenant), jobs in groups.items():
                try:
                    capture_group(page, origin, sid, tenant, jobs, skip_existing=not args.overwrite)
                except Exception as e:  # noqa: BLE001
                    print(f"FAIL {sid} tenant={tenant}: {e}")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
