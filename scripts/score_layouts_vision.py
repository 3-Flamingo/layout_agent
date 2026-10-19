"""Rescore layouts on a 0–100 scale.

DeepSeek-V4.1-Flash sees the review screenshot. Hosted Jev stays text-only:
awesome-jev documents that the hosted model does not accept images, and the
multimodal projects there are separate models, not a patch for this API.
Jev's ordered score is mapped linearly onto 0–100.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "external_scores"
T1_RENDERS = ROOT / "data" / "evalsys" / "annotate_t1" / "renders"
PAIR_RENDERS = ROOT / "data" / "evalsys" / "review_human_pairs" / "renders"
DEEPSEEK_MODEL = "deepseek-v4.1-flash"
JEV_MODEL = "jev-1.13.0"

JEV_CRITERIA = [
    "标注挤在一起或压在零件上，图面不能读",
    "大面积重叠、穿线，只能勉强辨认",
    "多处重叠或引线失控，阅读很费劲",
    "有明显重叠和拥挤，问题成片",
    "能读，但多处引线过长或互相干扰",
    "能读，有若干清楚的局部问题",
    "整体清楚，问题零散",
    "清楚，只剩个别小问题",
    "标注分开，引线短，几乎不压零件",
    "标注分开、引线短、不压零件、疏密合适",
]

_WRITE_LOCK = threading.Lock()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _done(path: Path) -> set[str]:
    return {row["id"] for row in _read_jsonl(path) if row.get("ok")}


def _append(path: Path, row: dict) -> None:
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_file(item: dict) -> Path | None:
    stem = Path(item["id"]).name
    if stem.endswith(".json"):
        stem = stem[: -len(".json")] + ".png"
    candidate = T1_RENDERS / stem
    if candidate.is_file():
        return candidate
    for pair in item.get("pair_sides") or []:
        image = PAIR_RENDERS / f"{pair.get('pair_id')}_{pair.get('side')}.png"
        if image.is_file():
            return image
    return None


def jpeg_data_url(path: Path, max_side: int = 1280) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _post(url: str, key: str, body: dict, timeout: int) -> dict:
    last_error = "request failed"
    for attempt in range(5):
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
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


def _parse_score(content: str) -> tuple[int, str]:
    import re

    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    if start >= 0:
        try:
            obj, _end = json.JSONDecoder().raw_decode(text[start:])
            score = int(round(float(obj["score"])))
            if 0 <= score <= 100:
                reason = obj.get("reason") or ""
                if not isinstance(reason, str):
                    reason = str(reason)
                return score, reason[:80]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    match = re.search(r'"score"\s*:\s*(\d{1,3})', text)
    if not match:
        raise ValueError("no json")
    score = int(match.group(1))
    if not 0 <= score <= 100:
        raise ValueError("score out of range")
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
    return score, (reason_match.group(1) if reason_match else "")[:80]


DEEPSEEK_TEXT = (
    "你在看一张工程图视图的标注布局截图。只根据这张图给布局质量打 0 到 100 的整数分。\n"
    "0 分：标注严重重叠或压在零件上，图不能读。\n"
    "100 分：标注分开、引线短、不压零件、疏密合适。\n"
    '只输出 JSON：{"score": 0到100的整数, "reason": "不超过30个字"}'
)


def score_deepseek(items: list[dict], workers: int, limit: int | None = None) -> None:
    key = os.environ["CHATANYWHERE_API_KEY"]
    path = OUT / "deepseek_vision.jsonl"
    done = _done(path)
    pending = [item for item in items if item["id"] not in done and render_file(item)]
    if limit is not None:
        pending = pending[:limit]
    print(f"deepseek vision pending {len(pending)} done {len(done)}", flush=True)
    counter = {"n": 0}
    lock = threading.Lock()

    def one(item: dict) -> None:
        image = render_file(item)
        started = time.time()
        try:
            if image is None:
                raise RuntimeError("missing render")
            body = {
                "model": DEEPSEEK_MODEL,
                "temperature": 0,
                "max_tokens": 300,
                "thinking": {"type": "disabled"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": jpeg_data_url(image)}},
                            {"type": "text", "text": DEEPSEEK_TEXT},
                        ],
                    }
                ],
            }
            data = _post("https://api.chatanywhere.tech/v1/chat/completions", key, body, 180)
            message = data["choices"][0]["message"]
            score, reason = _parse_score(message.get("content") or "")
            row = {
                "id": item["id"],
                "ok": True,
                "score": score,
                "reason": reason,
                "image": str(image.relative_to(ROOT).as_posix()),
                "model": data.get("model"),
                "elapsed_s": round(time.time() - started, 2),
                "usage": data.get("usage"),
            }
        except Exception as exc:  # noqa: BLE001
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


def score_jev(items: list[dict], workers: int, limit: int | None = None) -> None:
    key = os.environ["TYPESAFE_API_KEY"]
    sample = set(json.loads((OUT / "jev_sample_ids.json").read_text(encoding="utf-8"))["ids"])
    path = OUT / "jev_100pt.jsonl"
    done = _done(path)
    pending = [item for item in items if item["id"] in sample and item["id"] not in done]
    if limit is not None:
        pending = pending[:limit]
    print(f"jev pending {len(pending)} done {len(done)}", flush=True)

    def one(item: dict) -> None:
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
        started = time.time()
        try:
            data = _post("https://api.typesafe.ai/v1/systemone", key, body, 90)
            answer = data["answers"]["quality"]
            probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
            level = _jev_level(probabilities)
            expected = float(answer["score"])
            row = {
                "id": item["id"],
                "ok": True,
                "raw_score": expected,
                "level": level,
                "score_100": round(100.0 * expected / (len(JEV_CRITERIA) - 1), 1),
                "level_100": round(100.0 * level / (len(JEV_CRITERIA) - 1), 1),
                "confidence": answer.get("confidence"),
                "probabilities": probabilities,
                "model": data.get("model"),
                "input": "text",
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


def _ok_map(path: Path, field: str) -> dict[str, float]:
    found = {}
    for row in _read_jsonl(path):
        if row.get("ok") and field in row:
            found[row["id"]] = float(row[field])
    return found


def choose_threshold(items: list[dict], scores: dict[str, float], dest: Path | None = None) -> dict:
    perfect, defect = [], []
    for item in items:
        score = scores.get(item["id"])
        if score is None:
            continue
        for pair in item.get("pair_sides") or []:
            if pair.get("prefer") != "b" or pair.get("side") not in ("a", "b"):
                continue
            (perfect if pair["side"] == "b" else defect).append(score)
    best = None
    curve = []
    for threshold in range(0, 101):
        true_positive = sum(score >= threshold for score in perfect)
        false_positive = sum(score >= threshold for score in defect)
        true_rate = true_positive / len(perfect) if perfect else 0.0
        false_rate = false_positive / len(defect) if defect else 0.0
        youden = true_rate - false_rate
        predicted = true_positive + false_positive
        precision = true_positive / predicted if predicted else 0.0
        row = {
            "threshold": threshold,
            "youden": round(youden, 4),
            "precision": round(precision, 4),
            "perfect_kept": true_positive,
            "defect_kept": false_positive,
        }
        curve.append(row)
        rank = (youden, precision, threshold)
        if best is None or rank > best[0]:
            best = (rank, row)
    chosen = best[1]
    human = {grade: [] for grade in range(1, 6)}
    for item in items:
        grade = item.get("human_grade")
        score = scores.get(item["id"])
        if grade in human and score is not None:
            human[grade].append(score)
    human_summary = {}
    for grade, values in human.items():
        if not values:
            continue
        ordered = sorted(values)
        human_summary[str(grade)] = {
            "n": len(values),
            "median": ordered[len(ordered) // 2],
            "at_or_above_threshold": sum(value >= chosen["threshold"] for value in values),
        }
    above = sum(score >= chosen["threshold"] for score in scores.values())
    report = {
        "threshold": chosen["threshold"],
        "rule": "DeepSeek 分数大于等于该值，视为完美布局",
        "definition": "完美侧是 human_pairs 里标注为 prefer=b 的一侧，缺陷侧是 a",
        "selection": "在 0 到 100 里取约登指数最高的切分；并列时取得分更高、阈值更高的那个",
        "n_perfect_sides": len(perfect),
        "n_defect_sides": len(defect),
        "n_scored": len(scores),
        "n_at_or_above": above,
        "perfect_median": sorted(perfect)[len(perfect) // 2] if perfect else None,
        "defect_median": sorted(defect)[len(defect) // 2] if defect else None,
        "youden": chosen["youden"],
        "precision": chosen["precision"],
        "perfect_kept": chosen["perfect_kept"],
        "defect_kept": chosen["defect_kept"],
        "human_grade": human_summary,
        "score_hist": dict(sorted(Counter(int(score) // 10 * 10 for score in scores.values()).items())),
    }
    (dest or (OUT / "perfect_threshold.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        cursor = 0
        while cursor < len(values):
            end = cursor
            while end + 1 < len(values) and values[order[end + 1]] == values[order[cursor]]:
                end += 1
            average = (cursor + end) / 2 + 1
            for index in range(cursor, end + 1):
                result[order[index]] = average
            cursor = end + 1
        return result

    rx, ry = ranks(xs), ranks(ys)
    mean_x, mean_y = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry))
    return None if den == 0 else num / den


def compare(items: list[dict]) -> dict:
    deep = _ok_map(OUT / "deepseek_vision.jsonl", "score")
    jev = _ok_map(OUT / "jev_100pt.jsonl", "score_100")
    joined = [(deep[item_id], jev[item_id]) for item_id in jev if item_id in deep]
    deltas = [abs(left - right) for left, right in joined]
    report = {
        "deepseek_input": "screenshot",
        "jev_input": "text",
        "scale": "0-100",
        "n_deepseek": len(deep),
        "n_jev": len(jev),
        "n_compared": len(joined),
        "spearman": _spearman([pair[0] for pair in joined], [pair[1] for pair in joined]),
        "mean_abs_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "deepseek_mean_on_sample": (sum(pair[0] for pair in joined) / len(joined)) if joined else None,
        "jev_mean_on_sample": (sum(pair[1] for pair in joined) / len(joined)) if joined else None,
    }
    (OUT / "comparison_vision.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    import sys

    items = _read_jsonl(OUT / "corpus.jsonl")
    if "--probe" in sys.argv:
        score_deepseek(items, workers=1, limit=1)
        score_jev(items, workers=1, limit=1)
        return
    score_threads = []
    for target in (lambda: score_deepseek(items, workers=8), lambda: score_jev(items, workers=4)):
        thread = threading.Thread(target=target)
        thread.start()
        score_threads.append(thread)
    for thread in score_threads:
        thread.join()
    scores = _ok_map(OUT / "deepseek_vision.jsonl", "score")
    threshold = choose_threshold(items, scores)
    comparison = compare(items)
    print("THRESHOLD", json.dumps({k: threshold[k] for k in ("threshold", "youden", "precision", "n_at_or_above", "n_scored", "perfect_median", "defect_median", "perfect_kept", "defect_kept")}, ensure_ascii=False), flush=True)
    print("COMPARE", json.dumps(comparison, ensure_ascii=False), flush=True)
    from build_score_review import main as build_review

    build_review()


if __name__ == "__main__":
    main()
