#!/usr/bin/env python3
"""OSS 投图 key →（可选）do_dimension → output.json → Layout Agent 数据树.

数据流对齐：
  1) do_dimension/scripts/test_parametric.py：按 OSS key 下载 projectionInput
  2) ParametricTaskManager.projection → cache/output.json（projectionOutputs.viewList）
  3) 将各视图 dimensions / projectionLines 转为 LayoutState，写入 Agent manifest
     （加载 UX 参考 vdr_data_val：粘贴/选择 JSON 即入库）

用法示例：
  # 已有 output.json，直接入库
  python scripts/import_projection_pipeline.py --output-json E:/workspace/do_dimension/cache/output.json

  # 粘贴 OSS key：下载后若是 output 则直接转；若是 projectionInput 则尝试跑投图
  python scripts/import_projection_pipeline.py --oss-key root/6a6e95fd00e4848a4421bdc0e95248d685f0644f.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from layout_agent.config import load_config
from layout_agent.models import (
    ColorLegend,
    ColorLegendEntry,
    LayoutState,
    LinearAnno,
    Obstacle,
    Point,
    RadialAnno,
    RasterMeta,
)
from layout_agent.perceive import ACTIVE_PALETTE, Perceiver

import fetch_production_dataset as fpd

aligned_measurement = fpd.aligned_measurement
estimate_text = fpd.estimate_text
format_dim_value = fpd.format_dim_value
load_json_bytes = fpd.load_json_bytes

DEFAULT_OSS = "https://oss.private.designorder.cn/designorder/"
DEFAULT_DO_DIMENSION = Path(r"E:\workspace\do_dimension")


def _pt(v: Any) -> Point:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return Point(x=float(v[0]), y=float(v[1]))
    if isinstance(v, dict):
        return Point(x=float(v.get("x", 0)), y=float(v.get("y", 0)))
    return Point(x=0.0, y=0.0)


def detect_kind(data: Any) -> str:
    """Return projection_input | projection_output | cad_document | vdr_records | unknown."""
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            if "projectionOutputs" in first or (
                "viewList" in (first.get("projectionOutputs") or {})
            ):
                return "projection_output"
            if "image" in first and "drawing" in first:
                return "vdr_records"
            if "drawing" in first and isinstance(first.get("drawing"), dict):
                return "projection_output"
        return "unknown"
    if isinstance(data, dict):
        if "projectionInput" in data:
            return "projection_input"
        if "projectionOutputs" in data or "viewList" in data:
            return "projection_output"
        if "doc" in data and isinstance(data.get("doc"), list):
            return "cad_document"
        if "image" in data and "drawing" in data:
            return "vdr_records"
    return "unknown"


def download_oss_key(oss_prefix: str, key: str, dest: Path) -> Dict[str, Any] | List[Any]:
    key = key.strip().lstrip("/")
    if key.startswith("http://") or key.startswith("https://"):
        url = key
        # derive filename
        name = key.rstrip("/").split("/")[-1].split("?")[0] or "download.json"
    else:
        url = f"{oss_prefix.rstrip('/')}/{key}?t={int(datetime.now().timestamp() * 1000)}"
        name = Path(key).name
    dest = dest if dest.suffix else dest / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url}")
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        resp = client.get(url)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    data = load_json_bytes(resp.content)
    print(f"saved {dest} ({dest.stat().st_size} bytes) kind={detect_kind(data)}")
    return data


def normalize_output(data: Any) -> List[Dict[str, Any]]:
    """Normalize to list of {projectionOutputs: {...}} records."""
    if isinstance(data, list):
        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if "projectionOutputs" in item:
                out.append(item)
            elif "viewList" in item:
                out.append({"projectionOutputs": item})
            elif "drawing" in item and isinstance(item["drawing"], dict):
                # some dumps nest views under drawing
                dr = item["drawing"]
                if "viewList" in dr:
                    out.append({"projectionOutputs": dr, "enokiParams": item.get("enokiParams")})
        return out
    if isinstance(data, dict):
        if "projectionOutputs" in data:
            return [data]
        if "viewList" in data:
            return [{"projectionOutputs": data}]
    return []


def _view_obstacles(view: Dict[str, Any]) -> List[Obstacle]:
    obstacles: List[Obstacle] = []
    n = 0
    for i, pl in enumerate(view.get("projectionLines") or []):
        if not isinstance(pl, dict):
            continue
        feat = pl.get("feature") or {}
        if feat.get("basicType") == "HIDDEN_LINE" and pl.get("isProjectionVisible") is False:
            continue
        if pl.get("type") == "line" and pl.get("start") and pl.get("end"):
            n += 1
            obstacles.append(
                Obstacle(
                    id=f"pline_{view.get('uuid')}_{i}",
                    kind="polyline",
                    raw_type=str(feat.get("basicType") or "line"),
                    polyline=[_pt(pl["start"]), _pt(pl["end"])],
                    fill_role="edge",
                )
            )
        elif pl.get("type") == "polyline":
            verts = pl.get("vertexes") or pl.get("points") or []
            pts = [_pt(v) for v in verts]
            if len(pts) >= 2:
                n += 1
                obstacles.append(
                    Obstacle(
                        id=f"pline_{view.get('uuid')}_{i}",
                        kind="polyline",
                        raw_type=str(feat.get("basicType") or "polyline"),
                        polyline=pts,
                        fill_role="edge",
                    )
                )
    # geometry.viewloop: list of polygons, each polygon = list of [x,y]
    geom = view.get("geometry") or {}
    loop = geom.get("viewloop") or geom.get("viewLoop") or []
    if isinstance(loop, list) and loop:
        # flatten: either [[x,y],...] or [[[x,y],...], ...]
        polys: List[List[Any]] = []
        if loop and isinstance(loop[0], (list, tuple)) and loop[0] and isinstance(loop[0][0], (list, tuple)):
            polys = list(loop)
        elif loop and isinstance(loop[0], (list, tuple)):
            polys = [loop]
        for pi, poly in enumerate(polys):
            pts = [_pt(p) for p in poly if isinstance(p, (list, tuple, dict))]
            if len(pts) >= 2:
                obstacles.insert(
                    0,
                    Obstacle(
                        id=f"loop_{view.get('uuid')}_{pi}",
                        kind="polyline",
                        raw_type="view_loop",
                        polyline=pts,
                        fill_role="loop",
                    ),
                )
    # centerline-like dim entries typed as line
    for i, d in enumerate(view.get("dimensions") or []):
        if d.get("type") == "line" and d.get("start") and d.get("end"):
            obstacles.append(
                Obstacle(
                    id=f"dimline_{view.get('uuid')}_{i}",
                    kind="polyline",
                    raw_type="centerline",
                    polyline=[_pt(d["start"]), _pt(d["end"])],
                    fill_role="edge",
                )
            )
    return obstacles


def convert_projection_view(
    view: Dict[str, Any],
    *,
    solution_id: str,
    part_number: str,
    oss_key: str,
    palette: List[str],
    phase: str = "layout_after",
) -> Optional[LayoutState]:
    dims = [d for d in (view.get("dimensions") or []) if isinstance(d, dict)]
    active = [
        d
        for d in dims
        if d.get("type") in ("alignedDimension", "diameterDimension", "radialDimension", "diameter", "radial")
        and d.get("visible", True) is not False
    ]
    if not active and not any(d.get("type") == "alignedDimension" for d in dims):
        # skip empty / axonometric without dims
        if not any(d.get("type") == "alignedDimension" for d in dims):
            if str(view.get("viewType") or "").lower() in ("axonometric", "iso", "isometric"):
                return None

    linears: List[LinearAnno] = []
    diameters: List[RadialAnno] = []
    radials: List[RadialAnno] = []
    legend = ColorLegend(entries={})
    color_i = 0

    def next_color() -> str:
        nonlocal color_i
        c = palette[color_i % len(palette)]
        color_i += 1
        return c

    xs: List[float] = []
    ys: List[float] = []

    def expand(*pts: Point) -> None:
        for p in pts:
            xs.append(p.x)
            ys.append(p.y)

    for d in dims:
        if d.get("visible") is False:
            continue
        dtype = str(d.get("type") or "")
        uid = str((d.get("userData") or {}).get("uuid") or f"{dtype}_{len(linears)+len(diameters)}")
        if dtype == "alignedDimension":
            p1 = _pt(d.get("xLine1Point"))
            p2 = _pt(d.get("xLine2Point"))
            point = _pt(d.get("dimLinePoint"))
            text_point = _pt(d.get("textPosition") or d.get("dimLinePoint"))
            mid = Point(x=(p1.x + p2.x) / 2, y=(p1.y + p2.y) / 2)
            dx, dy = point.x - mid.x, point.y - mid.y
            norm = math.hypot(dx, dy) or 1.0
            try:
                ang = float(d["dimAngle"]) if d.get("dimAngle") is not None else None
            except (TypeError, ValueError):
                ang = None
            value = aligned_measurement(p1, p2, ang)
            content = format_dim_value(value, 2)
            color = next_color()
            text = estimate_text({}, content, 1.0)
            anno = LinearAnno(
                id=uid,
                color=color,
                p1=p1,
                p2=p2,
                dir=Point(x=dx / norm, y=dy / norm),
                point=point,
                value=value,
                text=text,
                dim_angle=ang,
                text_point=text_point,
            )
            linears.append(anno)
            legend.entries[color] = ColorLegendEntry(id=uid, category="linear")
            expand(p1, p2, point, text_point)
            tw, th = float(text["width"]), float(text["height"])
            expand(
                Point(x=text_point.x - tw / 2, y=text_point.y - th / 2),
                Point(x=text_point.x + tw / 2, y=text_point.y + th / 2),
            )
        elif dtype in ("diameterDimension", "diameter"):
            # best-effort: chord / farChord / text
            chord = _pt(d.get("chordPoint") or d.get("center") or [0, 0])
            far = _pt(d.get("farChordPoint") or d.get("chordPoint") or chord)
            text_pt = _pt(d.get("textPosition") or d.get("dimLinePoint") or far)
            anchor = Point(x=(chord.x + far.x) / 2, y=(chord.y + far.y) / 2)
            mid = Point(x=(anchor.x + text_pt.x) / 2, y=(anchor.y + text_pt.y) / 2)
            diam = d.get("diameter") or d.get("value")
            content = f"⌀{diam}" if diam is not None else "⌀"
            color = next_color()
            text = estimate_text({}, content, 1.0)
            anno = RadialAnno(
                id=uid,
                category="diameter",
                color=color,
                anchor=anchor,
                path=[anchor, mid, text_pt],
                text=text,
            )
            diameters.append(anno)
            legend.entries[color] = ColorLegendEntry(id=uid, category="diameter")
            expand(anchor, text_pt)
        elif dtype in ("radialDimension", "radial"):
            center = _pt(d.get("center") or d.get("chordPoint") or [0, 0])
            text_pt = _pt(d.get("textPosition") or d.get("dimLinePoint") or center)
            mid = Point(x=(center.x + text_pt.x) / 2, y=(center.y + text_pt.y) / 2)
            rad = d.get("radius") or d.get("value")
            content = f"R{rad}" if rad is not None else "R"
            color = next_color()
            text = estimate_text({}, content, 1.0)
            anno = RadialAnno(
                id=uid,
                category="radial",
                color=color,
                anchor=center,
                path=[center, mid, text_pt],
                text=text,
            )
            radials.append(anno)
            legend.entries[color] = ColorLegendEntry(id=uid, category="radial")
            expand(center, text_pt)

    if not linears and not diameters and not radials:
        return None

    obstacles = _view_obstacles(view)
    for ob in obstacles:
        for p in ob.polyline or []:
            expand(p)

    if not xs:
        return None
    pad = 15.0
    bbox_min = Point(x=min(xs) - pad, y=min(ys) - pad)
    bbox_max = Point(x=max(xs) + pad, y=max(ys) + pad)

    raw_uuid = str(view.get("uuid") or view.get("tag") or "unknown")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw_uuid)
    key_hash = hashlib.sha1(oss_key.encode("utf-8")).hexdigest()[:8] if oss_key else "local"
    sid = f"view_{safe}_{key_hash}"

    return LayoutState(
        state_id=sid,
        created_at=datetime.now(timezone.utc).isoformat(),
        source={
            "drawing_id": part_number or oss_key,
            "solution_id": solution_id,
            "solution_name": part_number or oss_key,
            "part_number": part_number,
            "tenant_id": "",
            "view_id": raw_uuid,
            "view_uuid": raw_uuid,
            "view_type": str(view.get("viewType") or ""),
            "origin": "do_dimension_output",
            "oss_key": oss_key,
        },
        meta={
            "text_height": 3.5,
            "revision": 0,
            "phase_tag": phase,
            "linear_count": len(linears),
            "diameter_count": len(diameters),
            "radial_count": len(radials),
            "obstacle_count": len(obstacles),
        },
        view={"bbox": {"min": bbox_min.model_dump(), "max": bbox_max.model_dump()}},
        raster=RasterMeta(
            instance_image="placeholder.png",
            width=1,
            height=1,
            origin=Point(x=0, y=0),
            pixel_size=0.25,
        ),
        color_legend=legend,
        obstacles=obstacles,
        linears=linears,
        diameters=diameters,
        radials=radials,
        linear_groups=[{"group_id": "default_linear", "member_ids": [x.id for x in linears]}],
    )


def run_parametric(
    oss_key: str,
    *,
    do_dimension_root: Path,
    python_exe: str,
    timeout_s: int = 1800,
) -> Path:
    """Invoke test_parametric.py with PARAMETRIC_INPUT_JSON; return cache/output.json."""
    script = do_dimension_root / "scripts" / "test_parametric.py"
    if not script.exists():
        raise FileNotFoundError(f"test_parametric.py not found: {script}")
    if not python_exe or not Path(python_exe).exists():
        raise RuntimeError(
            "未配置可用的投图 Python（paths.do_dimension_python）。\n"
            "当前 Agent 所用解释器通常没有 docore。\n"
            "请在 config/default.yaml 设置 do_dimension_python，"
            "例如 E:/miniforge/envs/layout_data_gen/python.exe"
        )
    # quick preflight
    probe = subprocess.run(
        [
            python_exe,
            "-c",
            "import sys; sys.path.insert(0, r'%s'); import docore; "
            "from dodimension.pooling.task.parametric_task_manager import ParametricTaskManager"
            % str(do_dimension_root).replace("\\", "\\\\"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"投图环境自检失败（{python_exe}）：\n"
            + ((probe.stderr or probe.stdout or "")[-2000:])
            + "\n请更换 paths.do_dimension_python 为含 docore 的环境。"
        )
    out_json = do_dimension_root / "cache" / "output.json"
    env = os.environ.copy()
    env["PARAMETRIC_INPUT_JSON"] = oss_key
    env["PARAMETRIC_SKIP_POST"] = "1"
    env["PARAMETRIC_EXIT_ON_ERROR"] = "1"
    env.setdefault("DOSERVICES_OSS_URL", "http://oss.private.designorder.cn/designorder/")
    print(f"running parametric: {python_exe} {script} key={oss_key}")
    proc = subprocess.run(
        [python_exe, str(script)],
        cwd=str(do_dimension_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "parametric failed\n"
            + (proc.stdout or "")[-3000:]
            + "\n"
            + (proc.stderr or "")[-3000:]
        )
    if not out_json.exists():
        raise FileNotFoundError(f"expected output missing: {out_json}")
    print(f"parametric ok → {out_json}")
    return out_json


def merge_states_into_dataset(
    states: List[LayoutState],
    *,
    dataset: Path,
    part_number: str,
    solution_id: str,
    phase: str,
    id_suffix: str,
) -> Dict[str, Any]:
    cfg = load_config()
    cfg.setdefault("perceive", {})["pixel_size"] = float(
        cfg.get("perceive", {}).get("pixel_size", 0.35)
    )
    perceiver = Perceiver(cfg)
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "snapshots").mkdir(parents=True, exist_ok=True)

    man_path = dataset / "manifest.json"
    if man_path.exists():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "dataset_name": dataset.name,
            "schema_version": "agent1.1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "owner": "layout_agent",
            "items": [],
            "part_numbers": [],
            "solution_id": solution_id,
        }

    added = []
    for st in states:
        case_id = f"{st.state_id}{id_suffix}" if id_suffix and not st.state_id.endswith(id_suffix) else st.state_id
        st.state_id = case_id
        st.meta = dict(st.meta or {})
        st.meta["phase_tag"] = phase
        rel_inst = f"snapshots/{case_id}.instance.png"
        rel_obs = f"snapshots/{case_id}.obstacle.png"
        rel_state = f"snapshots/{case_id}.state.json"
        rel_legend = f"snapshots/{case_id}.legend.json"
        print(f"  render {case_id} L{len(st.linears)} D{len(st.diameters)} R{len(st.radials)}")
        st = perceiver.render_icm(st, dataset / rel_inst, dataset / rel_obs)
        st.raster.instance_image = rel_inst
        st.raster.obstacle_image = rel_obs
        (dataset / rel_state).write_text(
            json.dumps(st.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (dataset / rel_legend).write_text(
            json.dumps(st.color_legend.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        item = {
            "state_path": rel_state,
            "instance_image_path": rel_inst,
            "obstacle_image_path": rel_obs,
            "legend_path": rel_legend,
            "tags": ["do_dimension", phase, str((st.source or {}).get("view_type") or "view"), part_number],
            "split": "prod",
            "solution_id": solution_id,
            "part_number": part_number,
            "view_uuid": (st.source or {}).get("view_uuid"),
            "view_type": (st.source or {}).get("view_type"),
            "phase": phase,
            "dim_count": len(st.linears) + len(st.diameters) + len(st.radials),
            "linear_count": len(st.linears),
            "diameter_count": len(st.diameters),
            "radial_count": len(st.radials),
        }
        added.append(item)

    drop_parts = {part_number}
    drop_paths = {i["state_path"] for i in added}
    kept = [
        i
        for i in (manifest.get("items") or [])
        if i.get("state_path") not in drop_paths and i.get("part_number") not in drop_parts
    ]
    kept.extend(added)
    manifest["items"] = kept
    parts = list(manifest.get("part_numbers") or [])
    if part_number and part_number not in parts:
        parts.append(part_number)
    manifest["part_numbers"] = parts
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["note"] = "Imported from do_dimension projection output (OSS key → output.json)"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"merged {len(added)} views → {man_path} (total {len(kept)})")
    return {"added": len(added), "remaining": len(kept), "part_number": part_number, "items": added}


def load_output_to_states(
    data: Any,
    *,
    solution_id: str,
    part_number: str,
    oss_key: str,
    phase: str,
) -> List[LayoutState]:
    records = normalize_output(data)
    if not records:
        raise ValueError("not a projection output.json (missing projectionOutputs.viewList)")
    palette = ACTIVE_PALETTE * 20
    states: List[LayoutState] = []
    for rec in records:
        po = rec.get("projectionOutputs") or {}
        for view in po.get("viewList") or []:
            st = convert_projection_view(
                view,
                solution_id=solution_id,
                part_number=part_number,
                oss_key=oss_key,
                palette=palette,
                phase=phase,
            )
            if st is not None:
                states.append(st)
    if not states:
        raise ValueError("no convertible views with dimensions in output")
    return states


def main() -> None:
    ap = argparse.ArgumentParser(description="OSS key / output.json → Layout Agent dataset")
    ap.add_argument("--oss-key", default="", help="e.g. root/6a6e95fd....json")
    ap.add_argument("--oss-prefix", default=DEFAULT_OSS)
    ap.add_argument("--output-json", default="", help="skip download/run; convert this output.json")
    ap.add_argument("--run-parametric", action="store_true", help="if key is projectionInput, run do_dimension")
    ap.add_argument("--do-dimension-root", default=str(DEFAULT_DO_DIMENSION))
    ap.add_argument(
        "--parametric-python",
        default=os.environ.get("DO_DIMENSION_PYTHON", ""),
        help="python exe that can import dodimension (OCC env)",
    )
    ap.add_argument("--merge-into", default="", help="dataset dir; default = config paths.dataset_root")
    ap.add_argument("--part-number", default="")
    ap.add_argument("--solution-id", default="")
    ap.add_argument("--part-suffix", default="·布局后")
    ap.add_argument("--id-suffix", default="")
    ap.add_argument("--phase", default="layout_after")
    args = ap.parse_args()

    cfg = load_config()
    if not args.parametric_python:
        args.parametric_python = str(cfg.get("paths", {}).get("do_dimension_python") or "")
    if not args.do_dimension_root or args.do_dimension_root == str(DEFAULT_DO_DIMENSION):
        cfg_root = cfg.get("paths", {}).get("do_dimension_root")
        if cfg_root:
            args.do_dimension_root = str(cfg_root)
    dataset = Path(args.merge_into) if args.merge_into else Path(
        cfg.get("paths", {}).get("dataset_root", "data/datasets/prod_BANJINJIAN_bad")
    )
    if not dataset.is_absolute():
        dataset = ROOT / dataset

    cache = ROOT / "data" / "raw" / "projection_imports"
    cache.mkdir(parents=True, exist_ok=True)

    oss_key = args.oss_key.strip()
    data: Any = None
    output_path: Optional[Path] = Path(args.output_json) if args.output_json else None

    if output_path:
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        data = load_json_bytes(output_path.read_bytes())
        oss_key = oss_key or output_path.name
    elif oss_key:
        dest = cache / Path(oss_key).name
        data = download_oss_key(args.oss_prefix, oss_key, dest)
        kind = detect_kind(data)
        if kind == "projection_input":
            # keep input for audit
            (cache / f"input_{Path(oss_key).name}").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
            local_out = cache / f"output_{Path(oss_key).stem}.json"
            do_out = Path(args.do_dimension_root) / "cache" / "output.json"
            if args.run_parametric:
                py = args.parametric_python
                if not py:
                    raise SystemExit(
                        "勾选了跑投图，但未配置 paths.do_dimension_python。"
                        "已在 default.yaml 示例中写入 layout_data_gen 路径。"
                    )
                output_path = run_parametric(
                    oss_key,
                    do_dimension_root=Path(args.do_dimension_root),
                    python_exe=py,
                )
                shutil.copy2(output_path, local_out)
                data = load_json_bytes(local_out.read_bytes())
            elif local_out.exists():
                print(f"projectionInput detected; reusing cached {local_out}")
                data = load_json_bytes(local_out.read_bytes())
            else:
                hint = (
                    "下载到的是【投图输入】projectionInput，还不能直接进 Agent。\n"
                    "请任选其一：\n"
                    "1) 勾选「跑投图」（需在 config 配置 paths.do_dimension_python 为 OCC 环境）\n"
                    "2) 先在 do_dimension 生成 cache/output.json，再在本页上传该文件\n"
                    f"3) 命令行：python scripts/import_projection_pipeline.py --output-json {do_out}"
                )
                print(hint, file=sys.stderr)
                raise SystemExit(2)
        elif kind == "cad_document":
            raise SystemExit(
                "this key is a CadDocument web doc; use scripts/fetch_production_dataset.py --local-doc instead"
            )
        elif kind == "vdr_records":
            raise SystemExit(
                "this JSON is VDR viewer schema (image+drawing boxes); "
                "Agent needs do_dimension output.json with projectionOutputs.viewList"
            )
        elif kind != "projection_output":
            raise SystemExit(f"unrecognized JSON kind={kind}")
    else:
        raise SystemExit("provide --oss-key or --output-json")

    stem = Path(oss_key).stem if oss_key else "import"
    part = args.part_number or f"proj_{stem[:12]}"
    if args.part_suffix and not part.endswith(args.part_suffix):
        part = f"{part}{args.part_suffix}"
    solution_id = args.solution_id or f"oss_{stem[:16]}"

    states = load_output_to_states(
        data,
        solution_id=solution_id,
        part_number=part,
        oss_key=oss_key,
        phase=args.phase,
    )
    print(f"converted {len(states)} views")
    result = merge_states_into_dataset(
        states,
        dataset=dataset,
        part_number=part,
        solution_id=solution_id,
        phase=args.phase,
        id_suffix=args.id_suffix,
    )
    print(json.dumps({k: result[k] for k in ("added", "remaining", "part_number")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
