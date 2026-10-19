"""Build a local HTML page that shows each scored layout and its grades."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "external_scores"
T1_RENDERS = ROOT / "data" / "evalsys" / "annotate_t1" / "renders"
PAIR_RENDERS = ROOT / "data" / "evalsys" / "review_human_pairs" / "renders"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _ok_by_id(path: Path) -> dict[str, dict]:
    found = {}
    if not path.exists():
        return found
    for row in _read_jsonl(path):
        if row.get("ok"):
            found[row["id"]] = row
    return found


def _rel_to_page(path: Path) -> str:
    return Path("..") / ".." / path.relative_to(ROOT).as_posix()


def _render_for(item: dict) -> str | None:
    stem = Path(item["id"]).name
    if stem.endswith(".json"):
        stem = stem[: -len(".json")] + ".png"
    t1 = T1_RENDERS / stem
    if t1.is_file():
        return _rel_to_page(t1).as_posix()
    for pair in item.get("pair_sides") or []:
        name = f"{pair.get('pair_id')}_{pair.get('side')}.png"
        candidate = PAIR_RENDERS / name
        if candidate.is_file():
            return _rel_to_page(candidate).as_posix()
    return None


def _threshold_file(name: str) -> float | None:
    path = OUT / name
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get("threshold")
    return None if value is None else float(value)


def _probs(row: dict | None) -> list[float] | None:
    if not row or not isinstance(row.get("probabilities"), dict):
        return None
    keys = sorted(row["probabilities"], key=lambda key: int(key))
    return [round(float(row["probabilities"][key]), 3) for key in keys]


def build_items() -> list[dict]:
    corpus = _read_jsonl(OUT / "corpus.jsonl")
    text_image = _ok_by_id(OUT / "deepseek_text_image.jsonl")
    triple = _ok_by_id(OUT / "deepseek_text_json_image.jsonl")
    image_only = _ok_by_id(OUT / "deepseek_vision.jsonl")
    jev_json = _ok_by_id(OUT / "jev_text_json.jsonl")
    jev_text = _ok_by_id(OUT / "jev_text_100pt.jsonl")
    deepseek_grade = _ok_by_id(OUT / "deepseek.jsonl")
    residual = _ok_by_id(OUT / "residual.jsonl")
    threshold = _threshold_file("perfect_threshold_text_image.json")
    triple_threshold = _threshold_file("perfect_threshold_triple.json")
    sample = set(json.loads((OUT / "jev_sample_ids.json").read_text(encoding="utf-8"))["ids"])
    rows = []
    for item in corpus:
        deep = text_image.get(item["id"])
        triple_row = triple.get(item["id"])
        image_row = image_only.get(item["id"])
        jev_row = jev_json.get(item["id"])
        residual_row = residual.get(item["id"])
        jev_text_row = jev_text.get(item["id"])
        grade_row = deepseek_grade.get(item["id"])
        name = Path(item["id"]).name
        pair = (item.get("pair_sides") or [{}])[0]
        deep_score = None if deep is None else int(deep["score"])
        triple_score = None if triple_row is None else int(triple_row["score"])
        image_score = None if image_row is None else int(image_row["score"])
        jev_score = None if jev_row is None else float(jev_row["score_100"])
        jev_text_score = None if jev_text_row is None else float(jev_text_row["score_100"])
        rows.append(
            {
                "id": item["id"],
                "name": name,
                "render": _render_for(item),
                "deepseek": deep_score,
                "reason": (deep or {}).get("reason") or "",
                "triple": triple_score,
                "triple_reason": (triple_row or {}).get("reason") or "",
                "image_only": image_score,
                "perfect": threshold is not None and deep_score is not None and deep_score >= threshold,
                "perfect_triple": triple_threshold is not None and triple_score is not None and triple_score >= triple_threshold,
                "jev": None if jev_score is None else round(jev_score),
                "jev_score": jev_score,
                "jev_text": None if jev_text_score is None else round(jev_text_score),
                "jev_text_score": jev_text_score,
                "grade5": None if grade_row is None else int(grade_row["grade"]),
                "jev_confidence": None if jev_row is None else jev_row.get("confidence"),
                "jev_probs": _probs(jev_row),
                "residual_score": None if residual_row is None else float(residual_row["score"]),
                "residual_display": None if residual_row is None else float(residual_row["display"]),
                "formula_score": None if residual_row is None else float(residual_row["formula_score"]),
                "residual_delta": None if residual_row is None else float(residual_row["residual"]),
                "human": item.get("human_grade"),
                "held_out": bool(item.get("held_out")),
                "origins": item.get("origins") or [],
                "in_jev_sample": item["id"] in sample,
                "state_text": item.get("state_text") or "",
                "pair_id": pair.get("pair_id"),
                "pair_side": pair.get("side"),
                "pair_split": pair.get("split"),
            }
        )
    return rows


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>布局打分对照</title>
<style>
  :root { color-scheme: light; --bg:#f4f1ea; --ink:#1c1915; --muted:#6b645c; --line:#ddd6cb; --card:#fffdf8; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.45 "Segoe UI", "PingFang SC", sans-serif; background:var(--bg); color:var(--ink); }
  header { position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line); padding:14px 20px 12px; }
  h1 { margin:0 0 4px; font-size:18px; font-weight:650; }
  .sub { color:var(--muted); margin:0 0 10px; }
  .filters { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  label { display:flex; gap:6px; align-items:center; color:var(--muted); }
  select, input[type="search"] { font:inherit; border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:6px; padding:6px 8px; }
  input[type="search"] { min-width:220px; }
  .count { margin-left:auto; color:var(--muted); }
  main { padding:16px 20px 40px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:12px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; cursor:pointer; }
  .card:hover { border-color:#b7ad9f; }
  .thumb { height:190px; background:#ece7df; display:flex; align-items:center; justify-content:center; }
  .thumb img { max-width:100%; max-height:190px; object-fit:contain; }
  .thumb .empty { color:var(--muted); font-size:12px; }
  .body { padding:10px 12px 12px; }
  .name { font-size:12px; color:var(--muted); word-break:break-all; }
  .grades { display:flex; gap:6px; margin:8px 0; flex-wrap:wrap; }
  .pill { font-size:12px; border-radius:999px; padding:2px 8px; background:#efeae2; }
  .g1 { background:#f3d0cb; } .g2 { background:#f6ddc4; } .g3 { background:#f3e7b8; } .g4 { background:#d5e4f5; } .g5 { background:#d7ead8; }
  .reason { margin:0; color:#3f3a34; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
  .pager { display:flex; gap:8px; justify-content:center; margin-top:16px; }
  button { font:inherit; border:1px solid var(--line); background:#fff; border-radius:6px; padding:6px 12px; cursor:pointer; }
  button:disabled { opacity:.4; cursor:default; }
  .modal { position:fixed; inset:0; background:rgba(28,25,21,.55); display:none; z-index:5; }
  .modal.open { display:flex; align-items:stretch; justify-content:center; padding:24px; }
  .sheet { background:var(--card); width:min(1100px, 100%); border-radius:12px; display:grid; grid-template-columns:minmax(0,1.2fr) minmax(280px,.8fr); overflow:hidden; }
  .preview { background:#ece7df; display:flex; align-items:center; justify-content:center; min-height:320px; }
  .preview img { max-width:100%; max-height:86vh; object-fit:contain; }
  .detail { padding:16px 18px; overflow:auto; }
  .detail h2 { margin:0 0 8px; font-size:16px; word-break:break-all; }
  pre { white-space:pre-wrap; background:#f6f3ed; border-radius:8px; padding:10px; font:12px/1.5 ui-monospace, Consolas, monospace; }
  .nav { display:flex; gap:8px; margin-top:12px; }
  @media (max-width:800px) { .sheet { grid-template-columns:1fr; } .count { margin-left:0; } }
</style>
</head>
<body>
<header>
  <h1>布局打分对照</h1>
  <p class="sub" id="blurb">文图是文字摘要加截图。三输入再加视图 JSON。Jev 是文字摘要加 JSON。</p>
  <div class="filters">
    <label>范围 <select id="scope">
      <option value="jev">Jev 抽到的 100 条</option>
      <option value="gap">文图和 Jev 分数不同</option>
      <option value="human">有人工分</option>
      <option value="perfect">文图达到完美线</option>
      <option value="all">全部 1574 条</option>
    </select></label>
    <label>文图 <select id="ds"></select></label>
    <label>Jev+JSON <select id="jv"></select></label>
    <label>人工 <select id="hu"></select></label>
    <label>排序 <select id="sort">
      <option value="gap">文图和 Jev 分差从大到小</option>
      <option value="scheme">文图和三输入分差从大到小</option>
      <option value="ds">文图从低到高</option>
      <option value="triple">三输入从低到高</option>
      <option value="jv">Jev+JSON 从低到高</option>
      <option value="jtext">Jev文 从低到高</option>
      <option value="hu">人工分从低到高</option>
      <option value="residual">残差从低到高</option>
    </select></label>
    <input id="q" type="search" placeholder="搜文件名、理由、摘要" />
    <span class="count" id="count"></span>
  </div>
</header>
<main>
  <div class="grid" id="grid"></div>
  <div class="pager">
    <button id="prev" type="button">上一页</button>
    <span id="page"></span>
    <button id="next" type="button">下一页</button>
  </div>
</main>
<div class="modal" id="modal">
  <div class="sheet">
    <div class="preview" id="preview"></div>
    <div class="detail">
      <h2 id="d-name"></h2>
      <div class="grades" id="d-grades"></div>
      <p id="d-reason"></p>
      <p id="d-meta" class="sub"></p>
      <div id="d-probs"></div>
      <pre id="d-state"></pre>
      <div class="nav">
        <button id="d-prev" type="button">上一张</button>
        <button id="d-next" type="button">下一张</button>
        <button id="d-close" type="button">关闭</button>
      </div>
    </div>
  </div>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const PACK = JSON.parse(document.getElementById("payload").textContent);
const ITEMS = PACK.items;
const THRESHOLD = PACK.threshold;
const TRIPLE_THRESHOLD = PACK.triple_threshold;
const PAGE = 24;
const $ = (id) => document.getElementById(id);
let page = 0;
let filtered = ITEMS;
let openAt = -1;

function fillSelect(el, options) {
  el.innerHTML = "";
  for (const [value, text] of options) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = text;
    el.appendChild(opt);
  }
}
fillSelect($("ds"), [["all","文图 · 全部"],["none","没有分"],["0-49","0–49"],["50-79","50–79"],["80-100","80–100"]]);
fillSelect($("jv"), [["all","Jev · 全部"],["none","没有分"],["0-49","0–49"],["50-79","50–79"],["80-100","80–100"]]);
fillSelect($("hu"), [["all","人工 · 全部"],["none","没有分"],["1","1 分"],["2","2 分"],["3","3 分"],["4","4 分"],["5","5 分"]]);
if (THRESHOLD != null) {
  let blurb = "文图是文字摘要加截图，达到 " + THRESHOLD + " 视为完美布局。三输入再加视图 JSON";
  blurb += TRIPLE_THRESHOLD == null ? "。" : "，它的完美线是 " + TRIPLE_THRESHOLD + "。";
  blurb += " Jev文是文字摘要，Jev+JSON 再加视图 JSON，两者都是全量 0–100。残差是公式加已训练残差，刻度和 0–100 不同。";
  $("blurb").textContent = blurb;
}

function matchBand(value, selected) {
  if (selected === "all") return true;
  if (selected === "none") return value == null;
  if (value == null) return false;
  if (selected === "0-49") return value < 50;
  if (selected === "50-79") return value >= 50 && value < 80;
  if (selected === "80-100") return value >= 80;
  return String(value) === selected;
}

function apply() {
  const scope = $("scope").value;
  const q = $("q").value.trim().toLowerCase();
  filtered = ITEMS.filter((item) => {
    if (scope === "jev" && !item.in_jev_sample) return false;
    if (scope === "gap" && !(item.jev != null && item.deepseek != null && item.jev !== item.deepseek)) return false;
    if (scope === "human" && item.human == null) return false;
    if (scope === "perfect" && !item.perfect) return false;
    if (!matchBand(item.deepseek, $("ds").value)) return false;
    if (!matchBand(item.jev, $("jv").value)) return false;
    if (!matchBand(item.human, $("hu").value)) return false;
    if (!q) return true;
    const hay = (item.name + " " + item.reason + " " + item.state_text + " " + (item.pair_id || "")).toLowerCase();
    return hay.includes(q);
  });
  const sort = $("sort").value;
  const num = (value) => value == null ? 99 : value;
  filtered.sort((a, b) => {
    if (sort === "gap") {
      const ga = a.jev == null || a.deepseek == null ? -1 : Math.abs(a.jev - a.deepseek);
      const gb = b.jev == null || b.deepseek == null ? -1 : Math.abs(b.jev - b.deepseek);
      return gb - ga || num(a.deepseek) - num(b.deepseek) || a.name.localeCompare(b.name);
    }
    if (sort === "scheme") {
      const ga = a.triple == null || a.deepseek == null ? -1 : Math.abs(a.triple - a.deepseek);
      const gb = b.triple == null || b.deepseek == null ? -1 : Math.abs(b.triple - b.deepseek);
      return gb - ga || num(a.deepseek) - num(b.deepseek) || a.name.localeCompare(b.name);
    }
    const key = sort === "ds" ? "deepseek" : sort === "triple" ? "triple" : sort === "jv" ? "jev" : sort === "jtext" ? "jev_text" : sort === "residual" ? "residual_score" : "human";
    return num(a[key]) - num(b[key]) || a.name.localeCompare(b.name);
  });
  page = 0;
  render();
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

function band(score) {
  if (score == null) return "pill";
  if (THRESHOLD != null && score >= THRESHOLD) return "pill g5";
  if (score >= 80) return "pill g4";
  if (score >= 50) return "pill g3";
  if (score >= 30) return "pill g2";
  return "pill g1";
}

function pill(label, score) {
  const text = score == null ? label + " —" : label + " " + score;
  if (label === "人工" || label === "五分") {
    const cls = score == null ? "pill" : "pill g" + score;
    return '<span class="' + cls + '">' + text + '</span>';
  }
  if (label === "残差") {
    const shown = score == null ? "残差 —" : "残差 " + Number(score).toFixed(1);
    return '<span class="pill">' + shown + '</span>';
  }
  const extra = label === "文图" && score != null && THRESHOLD != null && score >= THRESHOLD ? " 完美" : "";
  return '<span class="' + band(score) + '">' + text + extra + '</span>';
}

function render() {
  const start = page * PAGE;
  const slice = filtered.slice(start, start + PAGE);
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE));
  $("count").textContent = filtered.length + " 条";
  $("page").textContent = (page + 1) + " / " + pages;
  $("prev").disabled = page === 0;
  $("next").disabled = page + 1 >= pages;
  $("grid").innerHTML = slice.map((item, index) => {
    const img = item.render
      ? '<img alt="" loading="lazy" src="' + esc(item.render) + '" />'
      : '<span class="empty">没有预览图</span>';
    return '<article class="card" data-i="' + (start + index) + '"><div class="thumb">' + img + '</div><div class="body"><div class="name">' + esc(item.name) + '</div><div class="grades">'
      + pill("文图", item.deepseek) + pill("三输入", item.triple) + pill("残差", item.residual_score) + pill("Jev文", item.jev_text) + pill("Jev+JSON", item.jev) + pill("人工", item.human)
      + '</div><p class="reason">' + esc(item.reason || "没有理由") + '</p></div></article>';
  }).join("");
}

function show(index) {
  if (index < 0 || index >= filtered.length) return;
  openAt = index;
  const item = filtered[index];
  $("d-name").textContent = item.name;
  $("d-grades").innerHTML = pill("文图", item.deepseek) + pill("三输入", item.triple) + pill("残差", item.residual_score) + pill("Jev文", item.jev_text) + pill("Jev+JSON", item.jev) + pill("只看图", item.image_only) + pill("五分", item.grade5) + pill("人工", item.human)
    + (item.jev_score == null ? "" : '<span class="pill">Jev 期望 ' + Number(item.jev_score).toFixed(1) + '</span>');
  const reasons = [];
  if (item.reason) reasons.push("文图：" + item.reason);
  if (item.triple_reason) reasons.push("三输入：" + item.triple_reason);
  $("d-reason").textContent = reasons.join(" ") || "没有理由。";
  const tags = (item.origins || []).join(" · ");
  const residualBits = item.residual_score == null ? "" : ("公式 " + Number(item.formula_score).toFixed(1) + "  残差项 " + Number(item.residual_delta).toFixed(1) + "  裁到 0–100 后 " + Number(item.residual_display).toFixed(1));
  $("d-meta").textContent = [tags, item.held_out ? "冻结测试集" : "可训练来源", item.pair_id ? ("配对 " + item.pair_id + " 侧 " + item.pair_side) : "", residualBits, item.id].filter(Boolean).join("  ·  ");
  $("d-probs").textContent = item.jev_probs ? "Jev 从差到好的档位概率：" + item.jev_probs.join("  ") : "";
  $("d-state").textContent = item.state_text;
  $("preview").innerHTML = item.render ? '<img alt="" src="' + esc(item.render) + '" />' : '<span class="empty">没有预览图</span>';
  $("modal").classList.add("open");
}

$("grid").addEventListener("click", (event) => {
  const card = event.target.closest(".card");
  if (!card) return;
  show(Number(card.dataset.i));
});
["scope","ds","jv","hu","sort"].forEach((id) => $(id).addEventListener("change", apply));
$("q").addEventListener("input", apply);
$("prev").addEventListener("click", () => { page -= 1; render(); });
$("next").addEventListener("click", () => { page += 1; render(); });
$("d-close").addEventListener("click", () => $("modal").classList.remove("open"));
$("d-prev").addEventListener("click", () => show(openAt - 1));
$("d-next").addEventListener("click", () => show(openAt + 1));
$("modal").addEventListener("click", (event) => { if (event.target.id === "modal") $("modal").classList.remove("open"); });
document.addEventListener("keydown", (event) => {
  if (!$("modal").classList.contains("open")) return;
  if (event.key === "Escape") $("modal").classList.remove("open");
  if (event.key === "ArrowLeft") show(openAt - 1);
  if (event.key === "ArrowRight") show(openAt + 1);
});
apply();
if (new URLSearchParams(location.search).get("open") === "1") show(0);
</script>
</body>
</html>
"""


def main() -> None:
    items = build_items()
    payload = json.dumps(
        {
            "threshold": _threshold_file("perfect_threshold_text_image.json"),
            "triple_threshold": _threshold_file("perfect_threshold_triple.json"),
            "items": items,
        },
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    html = HTML.replace("__PAYLOAD__", payload)
    path = OUT / "review.html"
    path.write_text(html, encoding="utf-8")
    with_render = sum(1 for item in items if item["render"])
    jev_with_render = sum(1 for item in items if item["in_jev_sample"] and item["render"])
    print(f"wrote {path}")
    print(f"items {len(items)} with_render {with_render} jev_sample_with_render {jev_with_render}")


if __name__ == "__main__":
    main()
