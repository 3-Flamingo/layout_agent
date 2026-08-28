"""FastAPI app: Layout Agent demo UI + JSON APIs."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layout_agent.agent import LayoutAgent
from layout_agent.config import load_config
from layout_agent.perceive import Perceiver

app = FastAPI(title="Layout Agent Demo", version="0.1.0")
STATIC = Path(__file__).parent / "static"
STATIC.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def dataset_root() -> Path:
    cfg = load_config()
    p = Path(cfg.get("paths", {}).get("dataset_root", "data/datasets/demo"))
    return p if p.is_absolute() else ROOT / p


def _load_case_item(case_id: str):
    ds = dataset_root()
    manifest = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    for x in manifest["items"]:
        st_path = ds / x["state_path"]
        st = json.loads(st_path.read_text(encoding="utf-8"))
        if st.get("state_id") == case_id or case_id in x["state_path"]:
            return ds, x, st, manifest
    return ds, None, None, manifest


def _safe_under(ds: Path, rel: str) -> Path:
    target = (ds / rel).resolve()
    if not str(target).startswith(str(ds.resolve())):
        raise HTTPException(403, "path escape")
    return target


def _case_id_from_item(item: Dict[str, Any]) -> str:
    return Path(item["state_path"]).name.replace(".state.json", "")


def _delete_item_files(ds: Path, item: Dict[str, Any]) -> List[str]:
    removed: List[str] = []
    for key in ("state_path", "instance_image_path", "obstacle_image_path", "legend_path"):
        rel = item.get(key)
        if not rel:
            continue
        target = _safe_under(ds, rel)
        if target.exists() and target.is_file():
            target.unlink()
            removed.append(rel)
    case_id = _case_id_from_item(item)
    run_dir = (ds / "runs" / case_id).resolve()
    if run_dir.exists() and str(run_dir).startswith(str((ds / "runs").resolve())):
        shutil.rmtree(run_dir)
        removed.append(f"runs/{case_id}/")
    return removed


def _rebuild_part_numbers(manifest: Dict[str, Any]) -> None:
    parts: List[str] = []
    for item in manifest.get("items") or []:
        pn = item.get("part_number") or ""
        if pn and pn not in parts:
            parts.append(pn)
    manifest["part_numbers"] = parts
    if len(parts) == 1:
        manifest["part_number"] = parts[0]


def _write_manifest(ds: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = ds / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ds / "manifest.json")


def _reload_cases_payload(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "remaining": len(manifest.get("items") or []),
        "part_numbers": manifest.get("part_numbers") or [],
        "updated_at": manifest.get("updated_at"),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    html = STATIC / "index.html"
    if not html.exists():
        return HTMLResponse("<h1>missing static/index.html</h1>", status_code=500)
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    cfg = load_config()
    d = cfg.get("decide", {})
    return {
        "ok": True,
        "model": d.get("model"),
        "use_llm": d.get("use_llm"),
        "llm_configured": bool(d.get("api_key")),
        "dataset": str(dataset_root()),
    }


@app.get("/api/manifest")
def get_manifest():
    ds = dataset_root()
    path = ds / "manifest.json"
    if not path.exists():
        raise HTTPException(404, "manifest.json not found — run scripts/fetch_production_dataset.py")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    ds, item, state, _manifest = _load_case_item(case_id)
    if item is None or state is None:
        raise HTTPException(404, f"case {case_id} not found")

    run_dir = ds / "runs" / case_id
    run_result = None
    best_state = None
    if (run_dir / "result.json").exists():
        run_result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if (run_dir / "best.state.json").exists():
        best_state = json.loads((run_dir / "best.state.json").read_text(encoding="utf-8"))

    return {
        "item": item,
        "state": state,
        "best_state": best_state,
        "urls": {
            "instance": f"/api/file/{item['instance_image_path']}",
            "obstacle": f"/api/file/{item['obstacle_image_path']}" if item.get("obstacle_image_path") else None,
            "legend": f"/api/file/{item['legend_path']}",
            "best_instance": f"/api/run-file/{case_id}/best.instance.png",
            "result": f"/api/run-file/{case_id}/result.json",
        },
        "run": run_result,
    }


@app.delete("/api/cases/{case_id}")
def delete_case(case_id: str):
    """Delete one view (snapshot files + optional run dir) and update manifest."""
    ds, item, _state, manifest = _load_case_item(case_id)
    if item is None:
        raise HTTPException(404, f"case {case_id} not found")
    removed = _delete_item_files(ds, item)
    manifest["items"] = [
        x for x in manifest.get("items") or [] if _case_id_from_item(x) != case_id and case_id not in x.get("state_path", "")
    ]
    _rebuild_part_numbers(manifest)
    _write_manifest(ds, manifest)
    return {"deleted": case_id, "removed_files": removed, **_reload_cases_payload(manifest)}


@app.delete("/api/parts")
def delete_part(part_number: str, solution_id: Optional[str] = None):
    """Delete all views under a part node."""
    ds = dataset_root()
    man_path = ds / "manifest.json"
    if not man_path.exists():
        raise HTTPException(404, "manifest.json not found")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    victims = []
    kept = []
    for item in manifest.get("items") or []:
        pn = item.get("part_number") or manifest.get("part_number") or ""
        sid = str(item.get("solution_id") or manifest.get("solution_id") or "")
        match = pn == part_number and (solution_id is None or sid == str(solution_id))
        if match:
            victims.append(item)
        else:
            kept.append(item)
    if not victims:
        raise HTTPException(404, f"part {part_number!r} not found")
    removed: List[str] = []
    deleted_ids = []
    for item in victims:
        deleted_ids.append(_case_id_from_item(item))
        removed.extend(_delete_item_files(ds, item))
    manifest["items"] = kept
    _rebuild_part_numbers(manifest)
    _write_manifest(ds, manifest)
    return {
        "deleted_part": part_number,
        "deleted_cases": deleted_ids,
        "removed_files": removed,
        **_reload_cases_payload(manifest),
    }


@app.post("/api/import-oss")
async def import_oss(
    oss_key: str = Form(""),
    part_number: str = Form(""),
    part_suffix: str = Form("·布局后"),
    phase: str = Form("layout_after"),
    run_parametric: str = Form("0"),
    file: Optional[UploadFile] = File(None),
):
    """OSS key (projectionInput/output) or local output.json → data tree.

    Flow mirrors do_dimension test_parametric download + output.json load.
    """
    ds = dataset_root()
    cfg = load_config()
    prod = cfg.get("paths", {}).get("production") or {}
    oss_prefix = str(prod.get("oss_prefix") or "https://oss.private.designorder.cn/designorder/")
    do_root = str(cfg.get("paths", {}).get("do_dimension_root") or r"E:\workspace\do_dimension")
    do_py = str(cfg.get("paths", {}).get("do_dimension_python") or "")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "import_projection_pipeline.py"),
        "--oss-prefix",
        oss_prefix,
        "--merge-into",
        str(ds),
        "--part-suffix",
        part_suffix,
        "--phase",
        phase,
        "--do-dimension-root",
        do_root,
    ]
    if part_number.strip():
        cmd.extend(["--part-number", part_number.strip()])
    if do_py:
        cmd.extend(["--parametric-python", do_py])

    tmp_path: Optional[Path] = None
    try:
        if file is not None and file.filename:
            suffix = Path(file.filename).suffix or ".json"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = Path(tmp.name)
            cmd.extend(["--output-json", str(tmp_path)])
        elif oss_key.strip():
            cmd.extend(["--oss-key", oss_key.strip()])
            if run_parametric in ("1", "true", "yes", "on"):
                cmd.append("--run-parametric")
        else:
            raise HTTPException(400, "provide oss_key or output.json file")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT / 'src'};{ROOT / 'scripts'}" + (
            f";{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
        )
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2400,
            env=env,
        )
        if proc.returncode != 0:
            # Prefer stderr (pipeline prints user-facing hints there), then stdout.
            err_text = ((proc.stderr or "").strip() or (proc.stdout or "").strip())[-4000:]
            raise HTTPException(
                400,
                detail={
                    "message": err_text or "import-oss failed",
                    "stdout": (proc.stdout or "")[-5000:],
                    "stderr": (proc.stderr or "")[-5000:],
                },
            )
        manifest = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
        return {
            "ok": True,
            "log": (proc.stdout or "")[-4000:],
            **_reload_cases_payload(manifest),
        }
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@app.get("/api/file/{path:path}")
def get_file(path: str):
    ds = dataset_root()
    target = (ds / path).resolve()
    if not str(target).startswith(str(ds.resolve())):
        raise HTTPException(403, "path escape")
    if not target.exists():
        raise HTTPException(404, str(path))
    return FileResponse(target)


@app.get("/api/run-file/{case_id}/{name}")
def get_run_file(case_id: str, name: str):
    ds = dataset_root()
    run_dir = (ds / "runs" / case_id).resolve()
    target = (run_dir / name).resolve()
    if not str(target).startswith(str(run_dir)):
        raise HTTPException(403, "path escape")
    if not target.exists() and name.endswith(".png"):
        best = run_dir / "best.state.json"
        if best.exists():
            st = json.loads(best.read_text(encoding="utf-8"))
            img = st.get("raster", {}).get("instance_image")
            if img and Path(img).exists():
                return FileResponse(img)
        snap = ds / "snapshots" / f"{case_id}.instance.png"
        if snap.exists():
            return FileResponse(snap)
        raise HTTPException(404, name)
    if not target.exists():
        raise HTTPException(404, name)
    return FileResponse(target)


def _prepare_state(case_id: str):
    cfg = load_config()
    ds, item, _, _manifest = _load_case_item(case_id)
    if item is None:
        raise HTTPException(404, f"case {case_id} not found")
    perceiver = Perceiver(cfg)
    state = perceiver.load_state(ds / item["state_path"], ds)
    state.raster.instance_image = str((ds / item["instance_image_path"]).resolve())
    if item.get("obstacle_image_path"):
        state.raster.obstacle_image = str((ds / item["obstacle_image_path"]).resolve())
    return cfg, ds, state


@app.post("/api/cases/{case_id}/run")
def run_case(case_id: str):
    cfg, ds, state = _prepare_state(case_id)
    agent = LayoutAgent(cfg)
    result = agent.run(state, ds / "runs" / case_id)
    return result.model_dump(mode="json")


@app.get("/api/cases/{case_id}/run/stream")
def run_case_stream(case_id: str):
    """SSE: stream Agent progress events, final event contains full result."""
    cfg, ds, state = _prepare_state(case_id)
    run_dir = ds / "runs" / case_id
    q: queue.Queue = queue.Queue()

    def on_progress(evt: Dict[str, Any]) -> None:
        q.put(("progress", evt))

    def worker() -> None:
        try:
            agent = LayoutAgent(cfg)
            result = agent.run(state, run_dir, on_progress=on_progress)
            q.put(("done", result.model_dump(mode="json")))
        except Exception as e:
            q.put(("error", {"message": str(e)}))
        finally:
            q.put(("end", None))

    threading.Thread(target=worker, daemon=True).start()

    def event_gen() -> Generator[str, None, None]:
        while True:
            kind, payload = q.get()
            if kind == "end":
                break
            if kind == "progress":
                yield f"event: progress\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            elif kind == "done":
                yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            elif kind == "error":
                yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/runs")
def list_runs():
    ds = dataset_root()
    runs_dir = ds / "runs"
    if not runs_dir.exists():
        return {"runs": []}
    out = []
    for p in sorted(runs_dir.iterdir()):
        if not p.is_dir():
            continue
        r = p / "result.json"
        if r.exists():
            out.append({"case_id": p.name, "result": json.loads(r.read_text(encoding="utf-8"))})
    return {"runs": out}
