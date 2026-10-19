"""Score layout states with DeepSeek-V4.1-Flash and a 100-item Jev sample.

Keys are read from the environment and are never written to disk:
  CHATANYWHERE_API_KEY, TYPESAFE_API_KEY

Corpus: both sides of the real human-pair train and test sets, plus every
reviewed non-discard T1 layout. Synthetic dense pairs and discards are left out.
The prompt never includes the human grade.
"""
from __future__ import annotations

import json
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "external_scores"
SEED = 17
JEV_N = 100
DEEPSEEK_MODEL = "deepseek-v4.1-flash"
JEV_MODEL = "jev-1.13.0"

CRITERIA = [
    "标注严重重叠或压在零件上，图面不能读",
    "有多处重叠、穿线或引线失控",
    "能读，但有明显拥挤或引线问题",
    "整体清楚，只剩少量局部问题",
    "标注分开、引线短、不压零件",
]

_WRITE_LOCK = threading.Lock()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _xy(point) -> tuple[float, float] | None:
    if not isinstance(point, dict) or "x" not in point or "y" not in point:
        return None
    try:
        return float(point["x"]), float(point["y"])
    except (TypeError, ValueError):
        return None


def _dist_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    if length2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _text_aabb(cx: float, cy: float, width: float, height: float, angle: float) -> tuple[float, float, float, float] | None:
    if width <= 0 or height <= 0:
        return None
    rad = math.radians(angle)
    cosine, sine = abs(math.cos(rad)), abs(math.sin(rad))
    half_w, half_h = width / 2, height / 2
    extent_x = half_w * cosine + half_h * sine
    extent_y = half_w * sine + half_h * cosine
    return (cx - extent_x, cy - extent_y, cx + extent_x, cy + extent_y)


def _point_in_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            denom = yj - yi
            if abs(denom) > 1e-12 and x < (xj - xi) * (y - yi) / denom + xi:
                inside = not inside
        j = i
    return inside


def _segs_cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def _segment_hits_box(a: tuple[float, float], b: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    if max(a[0], b[0]) < x0 or min(a[0], b[0]) > x1 or max(a[1], b[1]) < y0 or min(a[1], b[1]) > y1:
        return False
    if x0 <= a[0] <= x1 and y0 <= a[1] <= y1:
        return True
    if x0 <= b[0] <= x1 and y0 <= b[1] <= y1:
        return True
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    edges = ((corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[3]), (corners[3], corners[0]))
    return any(_segs_cross(a, b, c, d) for c, d in edges)


def _polyline_segments(points: list) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    coords = [_xy(point) for point in points]
    coords = [point for point in coords if point]
    return list(zip(coords, coords[1:]))


def _largest_loop(obstacles: list) -> list[tuple[float, float]]:
    best: list[tuple[float, float]] = []
    best_area = 0.0
    for obstacle in obstacles:
        if not isinstance(obstacle, dict):
            continue
        if obstacle.get("fill_role") != "loop" and obstacle.get("raw_type") != "view_loop":
            continue
        poly = [_xy(point) for point in (obstacle.get("polyline") or [])]
        poly = [point for point in poly if point]
        if len(poly) < 3:
            continue
        area = 0.0
        for i, (x1, y1) in enumerate(poly):
            x2, y2 = poly[(i + 1) % len(poly)]
            area += x1 * y2 - x2 * y1
        area = abs(area) / 2
        if area > best_area:
            best, best_area = poly, area
    return best


def summarize_state(doc: dict, view_type: str | None) -> str:
    linears = list(doc.get("linears") or [])
    diameters = list(doc.get("diameters") or [])
    radials = list(doc.get("radials") or [])
    roughness = list(doc.get("roughness") or [])
    obstacles = list(doc.get("obstacles") or [])
    source = doc.get("source") if isinstance(doc.get("source"), dict) else {}
    view_type = view_type or source.get("view_type") or "unknown"

    bbox = (doc.get("view") or {}).get("bbox") if isinstance(doc.get("view"), dict) else None
    mn = _xy((bbox or {}).get("min")) if isinstance(bbox, dict) else None
    mx = _xy((bbox or {}).get("max")) if isinstance(bbox, dict) else None
    center = ((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2) if mn and mx else None
    loop = _largest_loop(obstacles)

    offsets: list[float] = []
    outside = 0
    on_part = 0
    sides = Counter()
    boxes: list[tuple[tuple[float, float, float, float], list]] = []
    dim_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def place(point: tuple[float, float], box, own_segments: list) -> None:
        nonlocal outside, on_part
        if mn and mx and not (mn[0] <= point[0] <= mx[0] and mn[1] <= point[1] <= mx[1]):
            outside += 1
        if loop and _point_in_poly(point[0], point[1], loop):
            on_part += 1
        if center is not None:
            dx, dy = point[0] - center[0], point[1] - center[1]
            if abs(dx) >= abs(dy):
                sides["右" if dx >= 0 else "左"] += 1
            else:
                sides["上" if dy >= 0 else "下"] += 1
        if box is not None:
            boxes.append((box, own_segments))

    for linear in linears:
        text = linear.get("text") if isinstance(linear.get("text"), dict) else {}
        point = _xy(linear.get("point") or linear.get("text_point"))
        p1, p2 = _xy(linear.get("p1")), _xy(linear.get("p2"))
        own = [(p1, p2)] if p1 and p2 else []
        dim_segments.extend(own)
        if point and p1 and p2:
            offsets.append(_dist_to_segment(point[0], point[1], p1[0], p1[1], p2[0], p2[1]))
        if point:
            place(
                point,
                _text_aabb(
                    point[0],
                    point[1],
                    float(text.get("width") or 0),
                    float(text.get("height") or 0),
                    float(linear.get("dim_angle") or 0),
                ),
                own,
            )

    for radial in diameters + radials:
        text = radial.get("text") if isinstance(radial.get("text"), dict) else {}
        raw_path = radial.get("path") or []
        if len(raw_path) > 40:
            raw_path = raw_path[:: max(1, math.ceil(len(raw_path) / 40))]
        path = [_xy(point) for point in raw_path]
        path = [point for point in path if point]
        anchor = _xy(radial.get("anchor"))
        end = path[-1] if path else anchor
        seq = path if len(path) >= 2 else ([anchor, end] if anchor and end and anchor != end else [])
        own = list(zip(seq, seq[1:]))
        dim_segments.extend(own)
        length = sum(math.hypot(stop[0] - start[0], stop[1] - start[1]) for start, stop in own)
        if own:
            offsets.append(length)
        if end:
            place(
                end,
                _text_aabb(end[0], end[1], float(text.get("width") or 0), float(text.get("height") or 0), 0.0),
                own,
            )

    part_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for obstacle in obstacles:
        if not isinstance(obstacle, dict):
            continue
        points = obstacle.get("polyline") or []
        if len(points) > 80:
            points = points[:: max(1, math.ceil(len(points) / 80))]
        part_segments.extend(_polyline_segments(points))

    overlaps = 0
    cross_part = 0
    cross_dim = 0
    for index, (box, own) in enumerate(boxes):
        own_ids = {id(seg) for seg in own}
        if any(_segment_hits_box(seg[0], seg[1], box) for seg in part_segments):
            cross_part += 1
        if any(_segment_hits_box(seg[0], seg[1], box) for seg in dim_segments if id(seg) not in own_ids):
            cross_dim += 1
        for other, _own in boxes[index + 1 :]:
            if box[2] > other[0] and other[2] > box[0] and box[3] > other[1] and other[3] > box[1]:
                overlaps += 1
    med = _median(offsets)
    n_long = sum(1 for value in offsets if value > 25.0)
    med_text = "无" if med is None else f"{med:.1f} mm"
    inside_text = str(on_part) if loop else "未知"
    return "\n".join(
        [
            f"视图: {view_type}",
            f"线性标注 {len(linears)}，直径 {len(diameters)}，半径 {len(radials)}，粗糙度 {len(roughness)}",
            f"文字框互相重叠 {overlaps} 对",
            f"文字框碰到零件轮廓 {cross_part} 条",
            f"文字框碰到其它尺寸线 {cross_dim} 条",
            f"文字中心在零件外轮廓内部 {inside_text} 条",
            f"文字中心在视图框外 {outside} 条",
            f"尺寸文字到被测边的距离：中位 {med_text}，大于 25 mm 的有 {n_long} 条",
            f"文字分布：左 {sides['左']}，右 {sides['右']}，上 {sides['上']}，下 {sides['下']}",
        ]
    )


def _blank_item(path: Path) -> dict:
    return {
        "id": _rel(path),
        "state_path": str(path),
        "origins": [],
        "view_keys": [],
        "view_type": None,
        "human_grade": None,
        "held_out": False,
        "held_out_reasons": [],
        "pair_sides": [],
    }


def _add_origin(item: dict, origin: str) -> None:
    if origin not in item["origins"]:
        item["origins"].append(origin)


def build_corpus() -> dict:
    items: dict[str, dict] = {}
    missing = []
    discarded = 0
    unlabeled = 0

    def item_for(path_str: str | None) -> dict | None:
        if not path_str:
            return None
        path = _resolve(path_str)
        if not path.is_file():
            missing.append(str(path))
            return None
        key = _rel(path)
        item = items.get(key)
        if item is None:
            item = _blank_item(path)
            items[key] = item
        return item

    gold_state_ids = set()
    gold_view_by_state = {}
    for row in _read_jsonl(ROOT / "data/evalsys/human_gold/t1_tasks.jsonl"):
        if not row.get("state"):
            continue
        state_id = _rel(_resolve(row["state"]))
        gold_state_ids.add(state_id)
        if row.get("view_key"):
            gold_view_by_state[state_id] = row["view_key"]
    gold_pairs = set()
    for row in _read_jsonl(ROOT / "data/evalsys/human_gold/t2_tasks.jsonl"):
        if row.get("id"):
            gold_pairs.add(row["id"])

    for row in _read_jsonl(ROOT / "data/evalsys/human_gold/exports/labels_annotate_t1.jsonl"):
        label = row.get("label") or {}
        if label.get("discard"):
            discarded += 1
            continue
        grade = label.get("grade")
        if grade is None:
            unlabeled += 1
            continue
        item = item_for(row.get("state"))
        if item is None:
            continue
        item["human_grade"] = int(grade)
        view_key = row.get("view_key") or gold_view_by_state.get(item["id"])
        if view_key and view_key not in item["view_keys"]:
            item["view_keys"].append(view_key)
        if item["id"] in gold_state_ids:
            _add_origin(item, "t1_gold")
            item["held_out"] = True
            if "t1_gold" not in item["held_out_reasons"]:
                item["held_out_reasons"].append("t1_gold")
        else:
            _add_origin(item, "t1_graded")

    for split, filename in (("train", "train.jsonl"), ("test", "test.jsonl")):
        for row in _read_jsonl(ROOT / "data/evalsys/human_pairs" / filename):
            for side in ("a", "b"):
                item = item_for(row.get(f"state_{side}"))
                if item is None:
                    continue
                _add_origin(item, f"hp_{split}")
                if row.get("view_key") and row["view_key"] not in item["view_keys"]:
                    item["view_keys"].append(row["view_key"])
                if not item["view_type"]:
                    item["view_type"] = row.get("view_type")
                item["pair_sides"].append(
                    {
                        "pair_id": row.get("pair_id"),
                        "side": side,
                        "split": split,
                        "prefer": (row.get("label") or {}).get("prefer"),
                        "pose_delta": row.get("pose_delta"),
                    }
                )
                if split == "test" or row.get("pair_id") in gold_pairs:
                    item["held_out"] = True
                    reason = "t2_gold" if row.get("pair_id") in gold_pairs else "hp_test"
                    if reason not in item["held_out_reasons"]:
                        item["held_out_reasons"].append(reason)

    summarized = []
    parse_errors = []
    for item in items.values():
        try:
            doc = json.loads(Path(item["state_path"]).read_text(encoding="utf-8"))
            item["state_text"] = summarize_state(doc, item.get("view_type"))
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            parse_errors.append({"id": item["id"], "error": type(exc).__name__})
            continue
        summarized.append(item)
    summarized.sort(key=lambda row: row["id"])

    OUT.mkdir(parents=True, exist_ok=True)
    corpus_path = OUT / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for item in summarized:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    rng = random.Random(SEED)
    sample_path = OUT / "jev_sample_ids.json"
    sample_ids = rng.sample([item["id"] for item in summarized], min(JEV_N, len(summarized)))
    if sample_path.exists():
        previous = json.loads(sample_path.read_text(encoding="utf-8")).get("ids") or []
        valid = [item_id for item_id in previous if item_id in {item["id"] for item in summarized}]
        if len(valid) == min(JEV_N, len(summarized)):
            sample_ids = valid
    sample_ids = sorted(sample_ids)
    (OUT / "jev_sample_ids.json").write_text(
        json.dumps({"seed": SEED, "ids": sample_ids}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    origin_counts = Counter()
    for item in summarized:
        for origin in item["origins"]:
            origin_counts[origin] += 1
    manifest = {
        "n_items": len(summarized),
        "n_jev_sample": len(sample_ids),
        "seed": SEED,
        "origin_counts": dict(origin_counts),
        "n_with_human_grade": sum(item["human_grade"] is not None for item in summarized),
        "n_held_out": sum(item["held_out"] for item in summarized),
        "excluded_t1_discards": discarded,
        "excluded_t1_unlabeled": unlabeled,
        "missing_state_files": len(missing),
        "parse_errors": len(parse_errors),
        "included": [
            "human_pairs/train.jsonl both sides",
            "human_pairs/test.jsonl both sides",
            "reviewed non-discard T1 grades, including frozen gold",
        ],
        "excluded": [
            "T1 discards",
            "synthetic train_with_synth extras",
            "cmu_perturb and cmu_perturb_dense",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "missing_states.json").write_text(
        json.dumps({"missing": missing[:200], "n_missing": len(missing), "parse_errors": parse_errors}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok"):
            done.add(row["id"])
    return done


def _append(path: Path, row: dict) -> None:
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _post(url: str, key: str, body: dict, timeout: int) -> dict:
    last_error = "request failed"
    for attempt in range(5):
        req = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise RuntimeError(last_error) from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = type(exc).__name__
            if attempt == 4:
                raise RuntimeError(last_error) from None
        time.sleep(1.5 * (2**attempt))
    raise RuntimeError(last_error)


def _deepseek_prompt(state_text: str) -> str:
    lines = "\n".join(f"{index + 1}. {text}" for index, text in enumerate(CRITERIA))
    return (
        "你是工程图标注布局评分员。只根据下面的布局统计打分，不要假设图里还有别的信息。\n"
        "五档从差到好，依次对应 1 到 5 分。\n"
        f"{lines}\n"
        '只输出 JSON：{"grade": 1到5的整数, "reason": "不超过30个字"}\n\n'
        f"布局统计：\n{state_text}"
    )


def _parse_grade(content: str) -> tuple[int, str]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    if start >= 0:
        try:
            obj, _end = json.JSONDecoder().raw_decode(text[start:])
            grade = int(obj["grade"])
            if 1 <= grade <= 5:
                reason = obj.get("reason") or ""
                if not isinstance(reason, str):
                    reason = str(reason)
                return grade, reason[:80]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    import re

    match = re.search(r'"grade"\s*:\s*([1-5])', text)
    if not match:
        raise ValueError("no json")
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
    return int(match.group(1)), (reason_match.group(1) if reason_match else "")[:80]


def score_deepseek(items: list[dict], workers: int) -> None:
    key = os.environ["CHATANYWHERE_API_KEY"]
    path = OUT / "deepseek.jsonl"
    done = _load_done(path)
    pending = [item for item in items if item["id"] not in done]
    print(f"deepseek pending {len(pending)} done {len(done)}", flush=True)
    counter = {"n": 0}
    lock = threading.Lock()

    def one(item: dict) -> None:
        body = {
            "model": DEEPSEEK_MODEL,
            "temperature": 0,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": _deepseek_prompt(item["state_text"])}],
        }
        started = time.time()
        try:
            data = _post("https://api.chatanywhere.tech/v1/chat/completions", key, body, 120)
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            try:
                grade, reason = _parse_grade(content)
            except (ValueError, json.JSONDecodeError, KeyError, TypeError):
                grade, reason = _parse_grade(message.get("reasoning_content") or "")
            row = {
                "id": item["id"],
                "ok": True,
                "grade": grade,
                "reason": reason,
                "model": data.get("model"),
                "elapsed_s": round(time.time() - started, 2),
                "usage": data.get("usage"),
            }
        except Exception as exc:  # noqa: BLE001 - keep the batch moving and record the row
            row = {"id": item["id"], "ok": False, "error": str(exc)[:400]}
        _append(path, row)
        with lock:
            counter["n"] += 1
            if counter["n"] % 25 == 0 or counter["n"] == len(pending):
                print(f"deepseek {counter['n']}/{len(pending)}", flush=True)

    if not pending:
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, item) for item in pending]
        for future in as_completed(futures):
            future.result()


def _jev_level(probabilities: dict) -> int:
    best_key, best_value = None, -1.0
    for key, value in probabilities.items():
        value = float(value)
        if best_key is None or value > best_value:
            best_key, best_value = str(key), value
    return int(best_key)


def score_jev(items: list[dict], workers: int) -> None:
    key = os.environ["TYPESAFE_API_KEY"]
    sample = set(json.loads((OUT / "jev_sample_ids.json").read_text(encoding="utf-8"))["ids"])
    chosen = [item for item in items if item["id"] in sample]
    path = OUT / "jev.jsonl"
    done = _load_done(path)
    pending = [item for item in chosen if item["id"] not in done]
    print(f"jev pending {len(pending)} done {len(done)}", flush=True)

    def one(item: dict) -> None:
        body = {
            "model": JEV_MODEL,
            "state": item["state_text"],
            "questions": {
                "quality": {
                    "type": "score",
                    "instructions": "给这张视图的标注布局质量分档",
                    "criteria": CRITERIA,
                }
            },
        }
        started = time.time()
        try:
            data = _post("https://api.typesafe.ai/v1/systemone", key, body, 90)
            answer = data["answers"]["quality"]
            probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
            level = _jev_level(probabilities)
            row = {
                "id": item["id"],
                "ok": True,
                "score": float(answer["score"]),
                "level": level,
                "grade": level + 1,
                "confidence": answer.get("confidence"),
                "probabilities": probabilities,
                "model": data.get("model"),
                "elapsed_s": round(time.time() - started, 2),
            }
        except Exception as exc:  # noqa: BLE001
            row = {"id": item["id"], "ok": False, "error": str(exc)[:400]}
        _append(path, row)

    if not pending:
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, item) for item in pending]
        for future in as_completed(futures):
            future.result()
    print(f"jev finished {len(pending)}", flush=True)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(values):
        end = cursor
        while end + 1 < len(values) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        average = (cursor + end) / 2 + 1
        for index in range(cursor, end + 1):
            ranks[order[index]] = average
        cursor = end + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry))
    if den == 0:
        return None
    return num / den


def _load_ok(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok"):
            rows[row["id"]] = row
    return rows


def compare(items: list[dict]) -> dict:
    deepseek = _load_ok(OUT / "deepseek.jsonl")
    jev = _load_ok(OUT / "jev.jsonl")
    by_id = {item["id"]: item for item in items}
    joined = []
    for item_id, jev_row in jev.items():
        deep_row = deepseek.get(item_id)
        if not deep_row:
            continue
        item = by_id[item_id]
        joined.append(
            {
                "id": item_id,
                "deepseek_grade": deep_row["grade"],
                "jev_grade": jev_row["grade"],
                "jev_score": jev_row["score"],
                "jev_confidence": jev_row.get("confidence"),
                "human_grade": item.get("human_grade"),
                "held_out": item.get("held_out"),
                "origins": item.get("origins"),
                "deepseek_reason": deep_row.get("reason"),
                "state_text": item.get("state_text"),
            }
        )
    deep_grades = [row["deepseek_grade"] for row in joined]
    jev_grades = [row["jev_grade"] for row in joined]
    jev_expected = [row["jev_score"] + 1 for row in joined]
    confusion = [[0 for _ in range(5)] for _ in range(5)]
    for row in joined:
        confusion[row["deepseek_grade"] - 1][row["jev_grade"] - 1] += 1
    deltas = [abs(a - b) for a, b in zip(deep_grades, jev_grades)]
    human_rows = [row for row in joined if row["human_grade"] is not None]
    deepseek_human = [
        (deepseek[item["id"]]["grade"], item["human_grade"])
        for item in items
        if item.get("human_grade") is not None and item["id"] in deepseek
    ]
    disagreements = sorted(joined, key=lambda row: abs(row["deepseek_grade"] - row["jev_grade"]), reverse=True)[:8]

    pair_stats = _pair_preference(items, deepseek, jev)
    report = {
        "n_corpus": len(items),
        "n_deepseek_ok": len(deepseek),
        "n_jev_ok": len(jev),
        "n_compared": len(joined),
        "spearman_discrete_grades": _spearman(deep_grades, jev_grades),
        "spearman_deepseek_vs_jev_expected": _spearman(deep_grades, jev_expected),
        "exact_agreement": (sum(1 for delta in deltas if delta == 0) / len(deltas)) if deltas else None,
        "within_one": (sum(1 for delta in deltas if delta <= 1) / len(deltas)) if deltas else None,
        "mean_abs_grade_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "deepseek_grade_hist": dict(sorted(Counter(deep_grades).items())),
        "jev_grade_hist": dict(sorted(Counter(jev_grades).items())),
        "confusion_deepseek_rows_jev_cols": confusion,
        "n_with_human_in_sample": len(human_rows),
        "spearman_deepseek_vs_human_in_sample": _spearman(
            [row["deepseek_grade"] for row in human_rows],
            [row["human_grade"] for row in human_rows],
        ),
        "spearman_jev_vs_human_in_sample": _spearman(
            [row["jev_grade"] for row in human_rows],
            [row["human_grade"] for row in human_rows],
        ),
        "n_deepseek_with_human_grade": len(deepseek_human),
        "spearman_deepseek_vs_human_all": _spearman(
            [pair[0] for pair in deepseek_human],
            [pair[1] for pair in deepseek_human],
        ),
        "pair_preference": pair_stats,
        "largest_disagreements": [
            {
                "id": row["id"],
                "deepseek_grade": row["deepseek_grade"],
                "jev_grade": row["jev_grade"],
                "jev_score": round(row["jev_score"], 3),
                "human_grade": row["human_grade"],
                "deepseek_reason": row["deepseek_reason"],
                "state_text": row["state_text"],
            }
            for row in disagreements
            if abs(row["deepseek_grade"] - row["jev_grade"]) > 0
        ],
    }
    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_joined(joined)
    return report


def _pair_preference(items: list[dict], deepseek: dict[str, dict], jev: dict[str, dict]) -> dict:
    sides: dict[tuple[str, str], dict[str, str]] = {}
    for item in items:
        for pair in item.get("pair_sides") or []:
            key = (pair.get("split") or "", pair.get("pair_id") or "")
            sides.setdefault(key, {})[pair["side"]] = item["id"]

    def tally(scores: dict[str, dict], field: str) -> dict:
        counts = Counter()
        for (split, _pair_id), pair_sides in sides.items():
            if "a" not in pair_sides or "b" not in pair_sides:
                continue
            left, right = scores.get(pair_sides["a"]), scores.get(pair_sides["b"])
            if not left or not right:
                continue
            counts["n"] += 1
            counts[split] += 1
            if right[field] > left[field]:
                counts[f"{split}_prefer_b"] += 1
            elif right[field] < left[field]:
                counts[f"{split}_prefer_a"] += 1
            else:
                counts[f"{split}_tie"] += 1
        return dict(counts)

    return {"deepseek_on_all_pairs": tally(deepseek, "grade"), "jev_on_sampled_pairs": tally(jev, "grade")}


def _write_joined(joined: list[dict]) -> None:
    path = OUT / "comparison_rows.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in joined:
            slim = {key: value for key, value in row.items() if key != "state_text"}
            handle.write(json.dumps(slim, ensure_ascii=False) + "\n")


def main() -> None:
    if not (OUT / "corpus.jsonl").exists():
        manifest = build_corpus()
        print("corpus", json.dumps(manifest, ensure_ascii=False), flush=True)
    items = _read_jsonl(OUT / "corpus.jsonl")
    workers = []
    for target in (lambda: score_deepseek(items, workers=8), lambda: score_jev(items, workers=4)):
        thread = threading.Thread(target=target)
        thread.start()
        workers.append(thread)
    for thread in workers:
        thread.join()
    report = compare(items)
    printable = {key: value for key, value in report.items() if key != "largest_disagreements"}
    print("REPORT", json.dumps(printable, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
