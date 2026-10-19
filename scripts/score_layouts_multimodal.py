"""Rescore on 0–100 with the inputs the image-only pass left out.

Scheme A, every layout: DeepSeek sees the screenshot and the text summary.
Scheme B, every layout: DeepSeek also sees the view JSON.
Scheme B, the same 100 ids: hosted Jev sees the text summary and the view JSON.
Hosted Jev still cannot take images.

The image-only files (deepseek_vision.jsonl, jev_100pt.jsonl, perfect_threshold.json)
are left in place. Keys come from CHATANYWHERE_API_KEY and TYPESAFE_API_KEY.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from score_layouts_vision import (
    JEV_CRITERIA,
    JEV_MODEL,
    OUT,
    ROOT,
    _append,
    _done,
    _jev_level,
    _parse_score,
    _post,
    _read_jsonl,
    choose_threshold,
    jpeg_data_url,
    render_file,
)

DEEPSEEK_MODEL = "deepseek-v4.1-flash"
JSON_CHAR_BUDGET = 100_000
JEV_JSON_CHAR_BUDGET = 48_000
TEXT_IMAGE_PATH = OUT / "deepseek_text_image.jsonl"
TRIPLE_PATH = OUT / "deepseek_text_json_image.jsonl"
JEV_PATH = OUT / "jev_text_json.jsonl"
JEV_TEXT_PATH = OUT / "jev_text_100pt.jsonl"

TEXT_IMAGE_INSTRUCTIONS = (
    "你在评估一张工程图视图的标注布局，分数是 0 到 100 的整数。\n"
    "你同时拿到两样东西，两样都要看：这张截图，以及下面由程序量出来的文字摘要。\n"
    "截图用来判断文字是否重叠、尺寸线是否交叉、文字是否压在零件上、图面是否拥挤。\n"
    "文字摘要里的毫米距离是量出来的：距离大表示尺寸文字离被测边远、引线长。"
    "重叠对数、碰轮廓条数也以摘要为准，并和截图互相核对。\n"
    "截图上看得出交叉或压线时，不要因为摘要里重叠对数是 0 就当成没有问题。\n"
    "摘要里距离很大时，不要因为截图被缩小就当成引线很短。\n"
    "0 分：标注严重重叠或压在零件上，图不能读。\n"
    "100 分：标注分开、引线短、不压零件、疏密合适。\n"
    '只输出 JSON：{"score": 0到100的整数, "reason": "不超过40个字"}'
)

TRIPLE_INSTRUCTIONS = (
    "你在评估一张工程图视图的标注布局，分数是 0 到 100 的整数。\n"
    "你同时拿到三样东西，三样都要看：截图、文字摘要、这张视图的 JSON。\n"
    "截图用来看重叠、交叉、压线和疏密。\n"
    "文字摘要给出重叠对数、碰轮廓条数，以及尺寸文字到被测边的毫米距离。\n"
    "JSON 的坐标单位是毫米，y 向上。linears、diameters、radials、roughness 是标注；"
    "text_point 是尺寸文字位置，point 是尺寸线位置，p1 和 p2 是被测几何。"
    "obstacles 是零件轮廓和投影线。\n"
    "三样不一致时：交叉和压线以截图为准，毫米距离以摘要和 JSON 为准。\n"
    "0 分：标注严重重叠或压在零件上，图不能读。\n"
    "100 分：标注分开、引线短、不压零件、疏密合适。\n"
    '只输出 JSON：{"score": 0到100的整数, "reason": "不超过40个字"}'
)


def _dump(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))


def _round_numbers(value, digits: int = 1):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round_numbers(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: _round_numbers(item, digits) for key, item in value.items()}
    return value


def _thin_polylines(doc: dict, max_points: int) -> None:
    obstacles = doc.get("obstacles")
    if not isinstance(obstacles, list):
        return
    for obstacle in obstacles:
        if not isinstance(obstacle, dict):
            continue
        poly = obstacle.get("polyline")
        if not isinstance(poly, list) or len(poly) <= max_points:
            continue
        if max_points <= 2:
            obstacle["polyline"] = [poly[0], poly[-1]]
            continue
        step = max(1, math.ceil((len(poly) - 1) / (max_points - 1)))
        kept = poly[::step]
        if kept[-1] is not poly[-1]:
            kept.append(poly[-1])
        obstacle["polyline"] = kept


def _slim_obstacles(doc: dict) -> None:
    meta = doc.get("meta")
    if isinstance(meta, dict):
        meta.pop("render_conflicts", None)
    doc.pop("raster", None)
    doc.pop("color_legend", None)
    obstacles = doc.get("obstacles")
    if not isinstance(obstacles, list):
        return
    slim = []
    for obstacle in obstacles:
        if not isinstance(obstacle, dict):
            continue
        points = []
        for point in obstacle.get("polyline") or []:
            if isinstance(point, dict) and "x" in point and "y" in point:
                points.append([round(float(point["x"])), round(float(point["y"]))])
            elif isinstance(point, list):
                points.append(point)
        slim.append(
            {
                "raw_type": obstacle.get("raw_type"),
                "fill_role": obstacle.get("fill_role"),
                "polyline": points,
            }
        )
    doc["obstacles"] = slim


def view_json_text(item: dict, budget: int = JSON_CHAR_BUDGET) -> tuple[str, dict]:
    path = Path(item["state_path"])
    raw = path.read_text(encoding="utf-8")
    doc = json.loads(raw)
    text = _dump(doc)
    meta = {
        "json_chars": len(text),
        "json_original_chars": len(raw),
        "json_compacted": False,
    }
    if len(text) <= budget:
        return text, meta
    doc = _round_numbers(doc, 1)
    for limit in (40, 16, 6, 2):
        _thin_polylines(doc, limit)
        text = _dump(doc)
        if len(text) <= budget:
            meta["json_chars"] = len(text)
            meta["json_compacted"] = True
            return text, meta
    _slim_obstacles(doc)
    text = _dump(doc)
    if len(text) > budget:
        obstacles = doc.get("obstacles") or []
        loops = []
        rest = []
        for obstacle in obstacles:
            if obstacle.get("raw_type") == "view_loop" or obstacle.get("fill_role") == "loop":
                loops.append(obstacle)
            else:
                rest.append(obstacle)
        stride = 2
        while len(text) > budget and stride <= 16:
            doc["obstacles"] = loops + rest[::stride]
            text = _dump(doc)
            stride *= 2
    meta["json_chars"] = len(text)
    meta["json_compacted"] = True
    return text, meta


def _summary_block(item: dict) -> str:
    return "文字摘要：\n" + item["state_text"]


def _json_block(text: str) -> str:
    return "视图 JSON：\n" + text


def _deepseek_body(content: list[dict]) -> dict:
    return {
        "model": DEEPSEEK_MODEL,
        "temperature": 0,
        "max_tokens": 400,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": content}],
    }


def _select(items: list[dict], path: Path, only_ids: set[str] | None, limit: int | None, need_render: bool) -> list[dict]:
    done = _done(path)
    pending = []
    for item in items:
        if item["id"] in done:
            continue
        if only_ids is not None and item["id"] not in only_ids:
            continue
        if need_render and render_file(item) is None:
            continue
        pending.append(item)
    if limit is not None:
        pending = pending[:limit]
    return pending


def _run(pending: list[dict], workers: int, one) -> None:
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, item) for item in pending]
        for future in as_completed(futures):
            future.result()


def score_text_image(items: list[dict], workers: int, only_ids: set[str] | None = None, limit: int | None = None) -> None:
    key = os.environ["CHATANYWHERE_API_KEY"]
    pending = _select(items, TEXT_IMAGE_PATH, only_ids, limit, need_render=True)
    print(f"text+image pending {len(pending)}", flush=True)
    counter = {"n": 0}
    lock = threading.Lock()

    def one(item: dict) -> None:
        image = render_file(item)
        try:
            if image is None:
                raise RuntimeError("missing render")
            content = [
                {"type": "image_url", "image_url": {"url": jpeg_data_url(image)}},
                {"type": "text", "text": TEXT_IMAGE_INSTRUCTIONS + "\n\n" + _summary_block(item)},
            ]
            data = _post("https://api.chatanywhere.tech/v1/chat/completions", key, _deepseek_body(content), 180)
            message = data["choices"][0]["message"]
            score, reason = _parse_score(message.get("content") or "")
            row = {
                "id": item["id"],
                "ok": True,
                "score": score,
                "reason": reason,
                "input": "text+image",
                "image": str(image.relative_to(ROOT).as_posix()),
                "model": data.get("model"),
                "usage": data.get("usage"),
            }
        except Exception as exc:  # noqa: BLE001
            row = {"id": item["id"], "ok": False, "error": str(exc)[:400], "input": "text+image"}
        _append(TEXT_IMAGE_PATH, row)
        with lock:
            counter["n"] += 1
            if counter["n"] % 25 == 0 or counter["n"] == len(pending):
                print(f"text+image {counter['n']}/{len(pending)}", flush=True)

    _run(pending, workers, one)


def score_triple(items: list[dict], workers: int, only_ids: set[str] | None = None, limit: int | None = None) -> None:
    key = os.environ["CHATANYWHERE_API_KEY"]
    pending = _select(items, TRIPLE_PATH, only_ids, limit, need_render=True)
    print(f"text+json+image pending {len(pending)}", flush=True)
    counter = {"n": 0}
    lock = threading.Lock()

    def one(item: dict) -> None:
        image = render_file(item)
        try:
            if image is None:
                raise RuntimeError("missing render")
            payload, meta = view_json_text(item)
            content = [
                {"type": "image_url", "image_url": {"url": jpeg_data_url(image)}},
                {
                    "type": "text",
                    "text": "\n\n".join(
                        [TRIPLE_INSTRUCTIONS, _summary_block(item), _json_block(payload)]
                    ),
                },
            ]
            data = _post("https://api.chatanywhere.tech/v1/chat/completions", key, _deepseek_body(content), 240)
            message = data["choices"][0]["message"]
            score, reason = _parse_score(message.get("content") or "")
            row = {
                "id": item["id"],
                "ok": True,
                "score": score,
                "reason": reason,
                "input": "text+json+image",
                "image": str(image.relative_to(ROOT).as_posix()),
                "model": data.get("model"),
                **meta,
                "usage": data.get("usage"),
            }
        except Exception as exc:  # noqa: BLE001
            row = {"id": item["id"], "ok": False, "error": str(exc)[:400], "input": "text+json+image"}
        _append(TRIPLE_PATH, row)
        with lock:
            counter["n"] += 1
            if counter["n"] % 25 == 0 or counter["n"] == len(pending):
                print(f"text+json+image {counter['n']}/{len(pending)}", flush=True)

    _run(pending, workers, one)


def _jev_quality_row(item_id: str, data: dict, *, input_name: str, meta: dict | None = None) -> dict:
    answer = data["answers"]["quality"]
    probabilities = {str(key): float(value) for key, value in answer["probabilities"].items()}
    level = _jev_level(probabilities)
    expected = float(answer["score"])
    row = {
        "id": item_id,
        "ok": True,
        "raw_score": expected,
        "level": level,
        "score_100": round(100.0 * expected / (len(JEV_CRITERIA) - 1), 1),
        "level_100": round(100.0 * level / (len(JEV_CRITERIA) - 1), 1),
        "confidence": answer.get("confidence"),
        "probabilities": probabilities,
        "model": data.get("model"),
        "input": input_name,
    }
    if meta:
        row.update(meta)
    return row


def score_jev_text(items: list[dict], workers: int, only_ids: set[str] | None = None, limit: int | None = None) -> None:
    key = os.environ["TYPESAFE_API_KEY"]
    pending = _select(items, JEV_TEXT_PATH, only_ids, limit, need_render=False)
    print(f"jev text pending {len(pending)}", flush=True)
    counter = {"n": 0}
    lock = threading.Lock()

    def one(item: dict) -> None:
        try:
            body = {
                "model": JEV_MODEL,
                "state": item["state_text"],
                "questions": {
                    "quality": {
                        "type": "score",
                        "instructions": "给这张视图的标注布局质量打分，从不能读到接近完美",
                        "criteria": JEV_CRITERIA,
                    }
                },
            }
            data = _post("https://api.typesafe.ai/v1/systemone", key, body, 90)
            row = _jev_quality_row(item["id"], data, input_name="text")
        except Exception as exc:  # noqa: BLE001
            row = {"id": item["id"], "ok": False, "error": str(exc)[:400], "input": "text"}
        _append(JEV_TEXT_PATH, row)
        with lock:
            counter["n"] += 1
            if counter["n"] % 25 == 0 or counter["n"] == len(pending):
                print(f"jev text {counter['n']}/{len(pending)}", flush=True)

    _run(pending, workers, one)


def score_jev(
    items: list[dict],
    workers: int,
    only_ids: set[str] | None = None,
    limit: int | None = None,
    full: bool = False,
) -> None:
    key = os.environ["TYPESAFE_API_KEY"]
    if full:
        chosen = items
    else:
        sample = set(json.loads((OUT / "jev_sample_ids.json").read_text(encoding="utf-8"))["ids"])
        chosen = [item for item in items if item["id"] in sample]
    pending = _select(chosen, JEV_PATH, only_ids, limit, need_render=False)
    print(f"jev text+json pending {len(pending)}", flush=True)
    counter = {"n": 0}
    lock = threading.Lock()

    def one(item: dict) -> None:
        try:
            payload, meta = view_json_text(item, budget=JEV_JSON_CHAR_BUDGET)
            state = _summary_block(item) + "\n\n" + _json_block(payload)
            body = {
                "model": JEV_MODEL,
                "state": state,
                "questions": {
                    "quality": {
                        "type": "score",
                        "instructions": "结合文字摘要和视图 JSON，给这张视图的标注布局质量打分，从不能读到接近完美",
                        "criteria": JEV_CRITERIA,
                    }
                },
            }
            data = _post("https://api.typesafe.ai/v1/systemone", key, body, 120)
            row = _jev_quality_row(item["id"], data, input_name="text+json", meta=meta)
        except Exception as exc:  # noqa: BLE001
            row = {"id": item["id"], "ok": False, "error": str(exc)[:400], "input": "text+json"}
        _append(JEV_PATH, row)
        with lock:
            counter["n"] += 1
            if counter["n"] % 25 == 0 or counter["n"] == len(pending):
                print(f"jev text+json {counter['n']}/{len(pending)}", flush=True)

    _run(pending, workers, one)


def _ok_scores(path: Path, field: str) -> dict[str, float]:
    found = {}
    for row in _read_jsonl(path):
        if row.get("ok") and field in row:
            found[row["id"]] = float(row[field])
    return found


def _probe_ids(items: list[dict]) -> tuple[str, str]:
    small = next(item["id"] for item in items if item["id"].endswith("view_0KkbgjF3_7265.state.json"))
    sample = set(json.loads((OUT / "jev_sample_ids.json").read_text(encoding="utf-8"))["ids"])
    largest = max(
        (item for item in items if item["id"] in sample),
        key=lambda item: Path(item["state_path"]).stat().st_size,
    )
    return small, largest["id"]


def _print_rows(ids: set[str]) -> None:
    lines = []
    for path in (TEXT_IMAGE_PATH, TRIPLE_PATH, JEV_PATH):
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            if row["id"] not in ids:
                continue
            slim = {key: row.get(key) for key in ("id", "ok", "score", "score_100", "reason", "input", "error", "json_chars", "json_compacted", "model")}
            slim["file"] = path.name
            slim["name"] = Path(row["id"]).name
            lines.append(slim)
    text = json.dumps(lines, ensure_ascii=False, indent=2)
    (OUT / "_probe_multimodal.json").write_text(text, encoding="utf-8")
    print(text, flush=True)


def check_json_budget(items: list[dict]) -> None:
    compacted = 0
    still_over = 0
    worst = 0
    for item in items:
        text, meta = view_json_text(item)
        worst = max(worst, meta["json_chars"])
        if meta["json_compacted"]:
            compacted += 1
        if len(text) > JSON_CHAR_BUDGET:
            still_over += 1
    print(json.dumps({"n": len(items), "compacted": compacted, "still_over": still_over, "worst_chars": worst}, ensure_ascii=False), flush=True)


def main() -> None:
    items = _read_jsonl(OUT / "corpus.jsonl")
    if "--check-json" in sys.argv:
        check_json_budget(items)
        return
    if "--probe" in sys.argv:
        small, large = _probe_ids(items)
        ids = {small, large}
        score_text_image(items, workers=1, only_ids=ids)
        score_triple(items, workers=1, only_ids=ids)
        score_jev(items, workers=1, only_ids=ids)
        _print_rows(ids)
        return

    if "--full-jev" in sys.argv:
        threads = [
            threading.Thread(target=score_jev_text, args=(items, 6)),
            threading.Thread(target=score_jev, args=(items, 4), kwargs={"full": True}),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        from build_score_review import main as build_review

        build_review()
        return

    threads = [
        threading.Thread(target=score_text_image, args=(items, 8)),
        threading.Thread(target=score_triple, args=(items, 6)),
        threading.Thread(target=score_jev, args=(items, 3)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    text_scores = _ok_scores(TEXT_IMAGE_PATH, "score")
    triple_scores = _ok_scores(TRIPLE_PATH, "score")
    text_report = choose_threshold(items, text_scores, dest=OUT / "perfect_threshold_text_image.json")
    triple_report = choose_threshold(items, triple_scores, dest=OUT / "perfect_threshold_triple.json")
    keys = ("threshold", "youden", "precision", "n_at_or_above", "n_scored", "perfect_median", "defect_median", "perfect_kept", "defect_kept")
    print("TEXT_IMAGE", json.dumps({key: text_report[key] for key in keys}, ensure_ascii=False), flush=True)
    print("TRIPLE", json.dumps({key: triple_report[key] for key in keys}, ensure_ascii=False), flush=True)

    from build_score_review import main as build_review

    build_review()


if __name__ == "__main__":
    main()
