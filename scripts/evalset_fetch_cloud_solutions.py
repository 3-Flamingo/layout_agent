#!/usr/bin/env python3
"""List and fetch additional production solutions from 公有云 + 中车.

Uses the same DoMist findByCondition API as drawing-2d, then converts each
solution with fetch_production_dataset.py.

Examples:
  python scripts/evalset_fetch_cloud_solutions.py --list-only
  python scripts/evalset_fetch_cloud_solutions.py --public 8 --zhongche 8
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import httpx

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "BD580877-32E0-1FF1-CE5E-B1EE94202857"

PUBLIC = {
    "name": "public",
    "tenant": "0",
    "base": "https://api.designorder.cn/designBackend",
    "oss": "https://designorder.oss-cn-shanghai.aliyuncs.com/",
    "drawing": "https://drawing.designorder.cn",
    # Same project folder as 070-3579, plus its parent (more drawings).
    "pids": ["2081917103046832128", "1476242051065106432"],
}
ZHONGCHE = {
    "name": "zhongche",
    "tenant": "250",
    "base": "https://api.private.designorder.cn/designBackend",
    "oss": "https://oss.private.designorder.cn/designorder/",
    "drawing": "https://v3.private.designorder.cn/drawing",
    # TEH project + BANJIN folder + shared 中车 ancestor.
    "pids": ["1514253342132609024", "1509555989437980672", "1453117211666137088"],
}


def _headers(tenant: str) -> Dict[str, str]:
    return {
        "x-access-token": TOKEN,
        "tenant-Id": tenant,
        "content-type": "application/json;charset=UTF-8",
    }


def list_solutions(cloud: Dict[str, str], page_size: int = 50, max_pages: int = 8) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    url = f"{cloud['base'].rstrip('/')}/doMistServer/esResource/findByCondition"
    for pid in cloud["pids"]:
        for page in range(1, max_pages + 1):
            body = {
                "includeAllChildren": True,
                "pid": pid,
                "resourceCode": "solution",
                "types": ["file"],
            }
            resp = httpx.post(
                url,
                params={"tenantId": cloud["tenant"], "pageNumber": page, "pageSize": page_size},
                json=body,
                headers=_headers(cloud["tenant"]),
                timeout=60.0,
            )
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("result") or []
            for row in rows:
                sid = str(row.get("id") or "")
                if sid:
                    seen[sid] = row
            total = int(payload.get("total") or 0)
            if page * page_size >= total or not rows:
                break
    return list(seen.values())


def _amount(row: Dict[str, Any]) -> int:
    raw = row.get("solutionDrawingAmount")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def pick(rows: List[Dict[str, Any]], already: Set[str], n: int) -> List[Dict[str, Any]]:
    usable = [
        r
        for r in rows
        if str(r.get("id") or "") not in already
        and r.get("solutionSolutionWebDocument")
    ]
    usable.sort(key=lambda r: (-_amount(r), str(r.get("name") or "")))
    return usable[:n]


def existing_solution_ids() -> Set[str]:
    ids: Set[str] = set()
    for p in (ROOT / "data" / "datasets").glob("*/manifest.json"):
        try:
            man = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        items = man.get("items") or []
        if not items:
            continue
        sid = str(man.get("solution_id") or "")
        if sid.isdigit():
            ids.add(sid)
    return ids


def safe_name(name: str, sid: str) -> str:
    slug = re.sub(r"[^\w.\-]+", "_", (name or "").strip())[:32].strip("_")
    return f"{slug or 'sol'}_{sid[-6:]}"


def fetch_one(cloud: Dict[str, str], row: Dict[str, Any], max_views: int) -> int:
    sid = str(row["id"])
    name = str(row.get("name") or sid)
    out = ROOT / "data" / "datasets" / f"prod_{safe_name(name, sid)}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "fetch_production_dataset.py"),
        "--solution-id",
        sid,
        "--tenant-id",
        cloud["tenant"],
        "--base-url",
        cloud["base"],
        "--oss-prefix",
        cloud["oss"],
        "--drawing-host",
        cloud["drawing"],
        "--max-views",
        str(max_views),
        "--view-types",
        "front,left,top",
        "--out",
        str(out),
        "--reuse-raw",
    ]
    print("RUN", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    return r.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--public", type=int, default=8, help="new 公有云 solutions to fetch")
    ap.add_argument("--zhongche", type=int, default=8, help="new 中车 solutions to fetch")
    ap.add_argument("--max-views", type=int, default=8)
    args = ap.parse_args()

    already = existing_solution_ids()
    print(f"already have {len(already)} solutions: {sorted(already)}")

    pub_rows = list_solutions(PUBLIC)
    zc_rows = list_solutions(ZHONGCHE)
    pub_docs = [r for r in pub_rows if r.get("solutionSolutionWebDocument")]
    zc_docs = [r for r in zc_rows if r.get("solutionSolutionWebDocument")]
    print(f"公有云 listed {len(pub_rows)} solutions, {len(pub_docs)} with web docs")
    print(f"中车 listed {len(zc_rows)} solutions, {len(zc_docs)} with web docs")

    pub_pick = pick(pub_rows, already, args.public)
    zc_pick = pick(zc_rows, already, args.zhongche)
    print("public pick:")
    for r in pub_pick:
        print(f"  {r.get('id')}  {r.get('name')}  drawings={_amount(r)}")
    print("zhongche pick:")
    for r in zc_pick:
        print(f"  {r.get('id')}  {r.get('name')}  drawings={_amount(r)}")

    if args.list_only:
        return

    failed = 0
    for cloud, rows in ((PUBLIC, pub_pick), (ZHONGCHE, zc_pick)):
        for row in rows:
            code = fetch_one(cloud, row, args.max_views)
            if code != 0:
                print(f"FAILED {row.get('id')} {row.get('name')} exit={code}")
                failed += 1
    if failed:
        raise SystemExit(f"{failed} solution fetches failed")
    print("done")


if __name__ == "__main__":
    main()
