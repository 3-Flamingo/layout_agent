"""Statistical figures for docs/布局优化工作总报告.md."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
SCORES = Path(__file__).resolve().parents[1] / "runs" / "external_scores"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimSun"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#D0D5DD",
        "axes.labelcolor": "#1F2933",
        "xtick.color": "#1F2933",
        "ytick.color": "#1F2933",
        "text.color": "#1F2933",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)

INK = "#1F2933"
LINE = "#E5E7EB"
C = {
    "formula": "#4C78A8",
    "residual": "#F58518",
    "jev": "#54A24B",
    "jev_json": "#72B7B2",
    "text_image": "#E45756",
    "triple": "#B279A2",
    "image": "#9D755D",
    "grade5": "#BAB0AC",
    "good": "#2E7D4F",
    "bad": "#C4504A",
    "tie": "#D6D3D1",
}


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style(ax) -> None:
    ax.grid(axis="y", color=LINE, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _last_ok(path: Path, value_key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok") and row.get(value_key) is not None:
            out[str(row["id"])] = float(row[value_key])
    return out


def _load() -> dict[str, dict[str, float]]:
    root = SCORES
    residual_score: dict[str, float] = {}
    formula: dict[str, float] = {}
    delta: dict[str, float] = {}
    for line in (root / "residual.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("ok"):
            continue
        residual_score[row["id"]] = float(row["score"])
        formula[row["id"]] = float(row["formula_score"])
        delta[row["id"]] = float(row["residual"])
    return {
        "formula": formula,
        "residual": residual_score,
        "delta": delta,
        "image": _last_ok(root / "deepseek_vision.jsonl", "score"),
        "text_image": _last_ok(root / "deepseek_text_image.jsonl", "score"),
        "triple": _last_ok(root / "deepseek_text_json_image.jsonl", "score"),
        "jev": _last_ok(root / "jev_text_100pt.jsonl", "score_100"),
        "jev_json": _last_ok(root / "jev_text_json.jsonl", "score_100"),
        "grade5": _last_ok(root / "deepseek.jsonl", "grade"),
    }


def residual_hists(data: dict[str, dict[str, float]]) -> None:
    panels = [
        ("公式分", list(data["formula"].values()), C["formula"], np.linspace(0, 100, 21)),
        ("残差项", list(data["delta"].values()), C["residual"], np.linspace(-40, 10, 26)),
        ("排序分", list(data["residual"].values()), "#E8A05A", np.linspace(-40, 110, 31)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    for ax, (title, values, color, bins) in zip(axes, panels):
        ax.hist(values, bins=bins, color=color, zorder=2)
        ax.axvline(float(np.median(values)), color=INK, linewidth=1, linestyle="--", zorder=3)
        ax.set_title(f"{title}  中位数 {np.median(values):.1f}")
        ax.set_ylabel("张数")
        _style(ax)
    fig.suptitle("全量 1574 张", fontsize=12, y=1.02)
    _save(fig, "residual_components.png")


def t1_display() -> None:
    grades = [1, 2, 3, 4, 5]
    formula = [9.1, 7.2, 8.8, 14.3, 34.1]
    residual = [6.7, 4.1, 2.0, 7.7, 27.0]
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.plot(grades, formula, marker="o", color=C["formula"], linewidth=2, label="公式")
    ax.plot(grades, residual, marker="o", color=C["residual"], linewidth=2, label="残差显示分")
    ax.set_xticks(grades)
    ax.set_xlabel("审核档次")
    ax.set_ylabel("0–100 均分")
    ax.set_title("冻结 136 张的百分制显示分")
    ax.set_ylim(0, 40)
    _style(ax)
    ax.legend(frameon=False)
    _save(fig, "t1_display_by_grade.png")


def residual_pairwise() -> None:
    labels = ["训练 578", "测试 170", "冻结 80"]
    residual = [91.9, 77.6, 86.3]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.bar(labels, residual, color=C["residual"], width=0.55, zorder=2)
    ax.axhline(76.3, color=C["formula"], linestyle="--", linewidth=1.2, label="公式在冻结 80 对：76.3%")
    ax.axhline(78.3, color="#B45309", linestyle=":", linewidth=1.1, label="成对门槛 78.3%")
    for bar, value in zip(bars, residual):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:g}%", ha="center", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("严格准确率（%）")
    ax.set_title("残差的成对严格准确率")
    _style(ax)
    ax.legend(frameon=False, loc="lower left")
    _save(fig, "residual_pairwise.png")


def visible_hard() -> None:
    grades = np.array([1, 2, 3, 4, 5])
    visible = [10, 6, 4, 1, 0]
    fail = [100, 100, 100, 98, 74]
    soft = [206, 231, 145, 169, 121]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axes[0].bar(grades, visible, color="#4C78A8", width=0.65, zorder=2)
    axes[0].set_title("可见硬约束条数中位数")
    axes[0].set_xlabel("审核档次")
    axes[0].set_ylabel("条数")
    axes[0].set_xticks(grades)
    _style(axes[0])
    ax2 = axes[0].twinx()
    ax2.plot(grades, fail, color="#C4504A", marker="o", linewidth=1.5)
    ax2.set_ylim(0, 120)
    ax2.set_ylabel("规则硬失败比例（%）", color="#C4504A")
    ax2.spines["top"].set_visible(False)
    axes[1].plot(grades, soft, marker="o", color="#72B7B2", linewidth=2)
    axes[1].set_title("软代价中位数")
    axes[1].set_xlabel("审核档次")
    axes[1].set_ylabel("软代价")
    axes[1].set_xticks(grades)
    _style(axes[1])
    _save(fig, "visible_hard_by_grade.png")


def violations() -> None:
    types = ["文字重叠", "穿尺寸线", "尺寸线相交", "蹭界线", "碰引线", "进零件", "压轮廓"]
    rows = {
        1: [2, 2, 7, 8, 4, 0, 0],
        2: [1, 1, 6, 18, 1, 1, 0.5],
        3: [0, 0, 5, 6, 0, 1, 0],
        4: [0, 0, 5, 4, 0, 0, 0],
        5: [0, 0, 2, 1, 0, 0, 0],
    }
    colors = ["#8C2F39", "#C4504A", "#E8A05A", "#4C78A8", "#54A24B"]
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    x = np.arange(len(types))
    width = 0.15
    for i, grade in enumerate(range(1, 6)):
        ax.bar(x + (i - 2) * width, rows[grade], width, color=colors[i], label=f"{grade} 档", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=20, ha="right")
    ax.set_ylabel("中位条数")
    ax.set_title("冻结 136 张各类违规的中位条数")
    _style(ax)
    ax.legend(frameon=False, ncol=5)
    _save(fig, "violation_medians.png")


def score_hists(data: dict[str, dict[str, float]]) -> None:
    panels = [
        ("只看图", data["image"], C["image"]),
        ("文图", data["text_image"], C["text_image"]),
        ("三输入", data["triple"], C["triple"]),
        ("Jev 文字", data["jev"], C["jev"]),
        ("Jev+JSON", data["jev_json"], C["jev_json"]),
        ("五分文字", data["grade5"], C["grade5"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2))
    for ax, (title, series, color) in zip(axes.ravel(), panels):
        values = list(series.values())
        if title == "五分文字":
            bins = np.arange(0.5, 6.5, 1)
        else:
            bins = np.arange(0, 105, 5)
        ax.hist(values, bins=bins, color=color, zorder=2)
        ax.set_title(f"{title}  均值 {np.mean(values):.1f}")
        _style(ax)
    fig.suptitle("全量分数分布", fontsize=12, y=1.01)
    _save(fig, "score_hists.png")


def human_medians() -> None:
    grades = [1, 2, 3, 4, 5]
    series = {
        "只看图": [78, 78, 78, 78, 78],
        "文图": [45, 45, 55, 60, 62],
        "三输入": [65, 78, 78, 78, 78],
    }
    colors = [C["image"], C["text_image"], C["triple"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for (name, values), color in zip(series.items(), colors):
        ax.plot(grades, values, marker="o", color=color, linewidth=2, label=name)
    ax.set_xticks(grades)
    ax.set_ylim(30, 100)
    ax.set_xlabel("审核档次（197 条）")
    ax.set_ylabel("该档分数中位数")
    ax.set_title("看图打分相对审核档次的中位数")
    _style(ax)
    ax.legend(frameon=False)
    _save(fig, "human_grade_medians.png")


def human_spearman() -> None:
    labels = ["Jev 文字", "三输入", "五分文字", "文图", "残差", "Jev+JSON", "只看图"]
    values = [0.58, 0.44, 0.32, 0.25, 0.24, 0.21, 0.17]
    colors = [C["jev"], C["triple"], C["grade5"], C["text_image"], C["residual"], C["jev_json"], C["image"]]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("与审核五档的 Spearman（197 条）")
    ax.set_title("百分制分数与审核五档的顺序相关")
    ax.grid(axis="x", color=LINE, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, value in enumerate(values):
        ax.text(value + 0.012, i, f"{value:.2f}", va="center", fontsize=9)
    _save(fig, "human_spearman.png")


def frozen_outcomes() -> None:
    labels = ["公式", "残差", "Jev 文字", "Jev+JSON", "文图", "三输入", "只看图"]
    agree = np.array([61, 69, 47, 57, 41, 23, 29])
    disagree = np.array([19, 11, 32, 21, 12, 8, 14])
    tie = np.array([0, 0, 1, 2, 27, 49, 37])
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, agree, color=C["good"], label="判对", zorder=2)
    ax.bar(x, disagree, bottom=agree, color=C["bad"], label="判错", zorder=2)
    ax.bar(x, tie, bottom=agree + disagree, color=C["tie"], label="同分", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("对数（共 80）")
    ax.set_ylim(0, 96)
    ax.set_title("冻结 80 对的判对、判错和同分")
    _style(ax)
    ax.legend(frameon=False, ncol=3)
    _save(fig, "frozen80_outcomes.png")


def pairwise() -> None:
    labels = ["残差", "Jev+JSON", "Jev 文字", "公式", "文图", "三输入", "只看图", "五分文字"]
    train = [91.9, 85.1, 80.3, np.nan, 59.3, 61.4, np.nan, 34.8]
    test = [77.6, 70.6, 60.6, np.nan, 42.4, 32.9, np.nan, 20.6]
    gold = [86.3, 71.3, 58.8, 76.3, 51.3, 28.8, 36.3, np.nan]
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    x = np.arange(len(labels))
    width = 0.26
    ax.bar(x - width, train, width, color="#8FB8DE", label="训练 578", zorder=2)
    ax.bar(x, test, width, color=C["formula"], label="测试 170", zorder=2)
    ax.bar(x + width, gold, width, color=C["residual"], label="冻结 80", zorder=2)
    ax.axhline(78.3, color="#B45309", linestyle="--", linewidth=1)
    ax.text(0.1, 80.2, "冻结 80 对门槛 78.3%", color="#B45309", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("严格准确率（%）")
    ax.set_ylim(0, 112)
    ax.set_title("成对严格准确率，同分算错")
    _style(ax)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _save(fig, "pairwise_accuracy.png")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def corr_heatmap(data: dict[str, dict[str, float]]) -> None:
    keys = ["formula", "residual", "jev", "jev_json", "text_image", "triple", "image"]
    labels = ["公式", "残差", "Jev 文字", "Jev+JSON", "文图", "三输入", "只看图"]
    ids = set.intersection(*(set(data[k]) for k in keys))
    mat = np.eye(len(keys))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i >= j:
                continue
            a = np.array([data[ki][k] for k in ids])
            b = np.array([data[kj][k] for k in ids])
            mat[i, j] = mat[j, i] = _spearman(a, b)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(mat, cmap="Blues", vmin=0.2, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_title(f"全量 Spearman（{len(ids)} 张都有分）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "score_corr.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("corr n", len(ids))
    print(np.round(mat, 3))


def perfect_split() -> None:
    labels = ["只看图 ≥75", "文图 ≥78", "三输入 ≥82"]
    perfect = [663 / 748 * 100, 448 / 748 * 100, 467 / 748 * 100]
    defect = [542 / 748 * 100, 298 / 748 * 100, 265 / 748 * 100]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, perfect, width, color=C["good"], label="完美侧留下", zorder=2)
    ax.bar(x + width / 2, defect, width, color=C["bad"], label="缺陷侧留下", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("留下比例（%）")
    ax.set_title("百分制切分后，两侧留下的比例")
    _style(ax)
    ax.legend(frameon=False)
    _save(fig, "perfect_split.png")


def proxy() -> None:
    labels = ["扰动验证", "同事挑战\n重画实例图", "同事挑战\n存图"]
    values = [56, 28, 30]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar(labels, values, color=C["grade5"], width=0.6, zorder=2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value}%", ha="center", fontsize=9)
    ax.set_ylim(0, 80)
    ax.set_ylabel("严格准确率（%）")
    ax.set_title("占位成对代理")
    _style(ax)
    _save(fig, "proxy_baseline.png")


def main() -> None:
    data = _load()
    residual_hists(data)
    t1_display()
    residual_pairwise()
    visible_hard()
    violations()
    score_hists(data)
    human_medians()
    human_spearman()
    frozen_outcomes()
    pairwise()
    corr_heatmap(data)
    perfect_split()
    proxy()
    print(OUT)


if __name__ == "__main__":
    main()
