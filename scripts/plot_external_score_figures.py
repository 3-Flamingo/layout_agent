"""Charts for docs/external_score_experiments.md. Numbers match that write-up."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
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
        "axes.titleweight": "medium",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

C = {
    "formula": "#4C78A8",
    "residual": "#F58518",
    "jev": "#54A24B",
    "jev_json": "#72B7B2",
    "text_image": "#E45756",
    "triple": "#B279A2",
    "image": "#9D755D",
    "grade5": "#BAB0AC",
    "ink": "#1F2933",
    "muted": "#6B7280",
    "line": "#D0D5DD",
    "good": "#2E7D4F",
    "bad": "#C4504A",
    "tie": "#D6D3D1",
}


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def gate() -> None:
    labels = ["公式", "残差", "Jev 文字", "Jev+JSON", "文图", "三输入", "只看图"]
    t1 = [0.511, 0.428, 0.509, 0.177, 0.207, 0.328, 0.243]
    t2 = [76.3, 86.3, 58.8, 71.3, 51.3, 28.8, 36.3]
    colors = [
        C["formula"],
        C["residual"],
        C["jev"],
        C["jev_json"],
        C["text_image"],
        C["triple"],
        C["image"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, values, thresh, title, ylab in (
        (axes[0], t1, 0.531, "T1 Spearman（136 张）", "Spearman"),
        (axes[1], t2, 78.3, "T2 严格准确率（80 对）", "严格准确率（%）"),
    ):
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, width=0.72, zorder=2)
        ax.axhline(thresh, color="#B45309", linestyle="--", linewidth=1.1, zorder=3)
        ax.text(
            len(labels) - 0.55,
            thresh,
            f" 门槛 {thresh:g}",
            va="bottom",
            ha="right",
            color="#B45309",
            fontsize=9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color=C["line"], zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:g}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=C["ink"],
            )
    axes[0].set_ylim(0, 0.72)
    axes[1].set_ylim(0, 108)
    fig.suptitle("冻结门禁：只有残差的 T2 越过门槛，没有一套两项都过", fontsize=13, y=1.02)
    _save(fig, "gate_t1_t2.png")


def grade_means() -> None:
    grades = [1, 2, 3, 4, 5]
    formula = [9.1, 7.2, 8.8, 14.3, 34.1]
    residual = [-4.2, -3.9, -5.7, 2.7, 24.3]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(grades, formula, marker="o", color=C["formula"], linewidth=2, label="公式")
    ax.plot(grades, residual, marker="o", color=C["residual"], linewidth=2, label="残差排序分")
    ax.axhline(0, color=C["line"], linewidth=1)
    ax.set_xticks(grades)
    ax.set_xlabel("人工档次")
    ax.set_ylabel("该档均分")
    ax.set_title("冻结 136 张：1–3 档没有随档次升高")
    ax.grid(axis="y", color=C["line"])
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)
    _save(fig, "t1_grade_means.png")


def distributions() -> None:
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 90]
    series = {
        "只看图": [0, 3, 29, 109, 123, 22, 1010, 274, 4],
        "文图": [0, 10, 19, 215, 134, 280, 647, 254, 15],
        "三输入": [1, 5, 11, 17, 16, 119, 676, 672, 57],
    }
    colors = [C["image"], C["text_image"], C["triple"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(bins))
    width = 0.26
    for i, (name, counts) in enumerate(series.items()):
        ax.bar(
            x + (i - 1) * width,
            counts,
            width=width,
            color=colors[i],
            label=name,
            zorder=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}" for b in bins])
    ax.set_xlabel("分数所在的十位")
    ax.set_ylabel("张数（共 1574）")
    ax.set_title("DeepSeek 百分制挤在 70–80 分")
    ax.grid(axis="y", color=C["line"], zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, ncol=3)
    _save(fig, "deepseek_deciles.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    grades = [1, 2, 3, 4, 5]
    counts = [7, 73, 725, 662, 107]
    ax.bar(grades, counts, color=C["grade5"], width=0.7, zorder=2)
    for g, n in zip(grades, counts):
        ax.text(g, n, str(n), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(grades)
    ax.set_xlabel("DeepSeek 五分")
    ax.set_ylabel("张数")
    ax.set_title("文字摘要五分：集中在 3 分和 4 分")
    ax.set_ylim(0, 860)
    ax.grid(axis="y", color=C["line"], zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _save(fig, "deepseek_five.png")


def human_medians() -> None:
    grades = [1, 2, 3, 4, 5]
    series = {
        "只看图": [78, 78, 78, 78, 78],
        "文图": [45, 45, 55, 60, 62],
        "三输入": [65, 78, 78, 78, 78],
    }
    colors = [C["image"], C["text_image"], C["triple"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for (name, values), color in zip(series.items(), colors):
        ax.plot(grades, values, marker="o", color=color, linewidth=2, label=name)
    ax.set_xticks(grades)
    ax.set_ylim(30, 100)
    ax.set_xlabel("人工档次（197 条）")
    ax.set_ylabel("该档分数中位数")
    ax.set_title("看图之后，人工档次几乎被打成同一个数")
    ax.grid(axis="y", color=C["line"])
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)
    _save(fig, "human_grade_medians.png")


def pairwise() -> None:
    labels = ["残差", "Jev+JSON", "Jev 文字", "公式", "文图", "三输入", "只看图", "五分文字"]
    train = [91.9, 85.1, 80.3, np.nan, 59.3, 61.4, np.nan, 34.8]
    test = [77.6, 70.6, 60.6, np.nan, 42.4, 32.9, np.nan, 20.6]
    gold = [86.3, 71.3, 58.8, 76.3, 51.3, 28.8, 36.3, np.nan]
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    x = np.arange(len(labels))
    width = 0.26
    ax.bar(x - width, train, width, color="#8FB8DE", label="训练 578", zorder=2)
    ax.bar(x, test, width, color=C["formula"], label="测试 170", zorder=2)
    ax.bar(x + width, gold, width, color=C["residual"], label="冻结 80", zorder=2)
    ax.axhline(78.3, color="#B45309", linestyle="--", linewidth=1, zorder=3)
    ax.text(len(labels) - 0.4, 79.2, "T2 门槛 78.3%", ha="right", color="#B45309", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("严格准确率（%）")
    ax.set_ylim(0, 112)
    ax.set_title("成对严格准确率：同分算错")
    ax.grid(axis="y", color=C["line"], zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _save(fig, "pairwise_accuracy.png")


def frozen_outcomes() -> None:
    labels = ["公式", "残差", "Jev 文字", "Jev+JSON", "文图", "三输入", "只看图"]
    agree = np.array([61, 69, 47, 57, 41, 23, 29])
    disagree = np.array([19, 11, 32, 21, 12, 8, 14])
    tie = np.array([0, 0, 1, 2, 27, 49, 37])
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    x = np.arange(len(labels))
    ax.bar(x, agree, color=C["good"], label="判对", zorder=2)
    ax.bar(x, disagree, bottom=agree, color=C["bad"], label="判错", zorder=2)
    ax.bar(x, tie, bottom=agree + disagree, color=C["tie"], label="同分", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("对数（共 80）")
    ax.set_ylim(0, 96)
    ax.set_title("冻结 80 对：DeepSeek 的同分把严格准确率压下去")
    ax.grid(axis="y", color=C["line"], zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, ncol=3)
    _save(fig, "frozen80_outcomes.png")


def human_spearman() -> None:
    labels = ["Jev 文字", "三输入", "五分文字", "文图", "残差", "Jev+JSON", "只看图"]
    values = [0.58, 0.44, 0.32, 0.25, 0.24, 0.21, 0.17]
    colors = [
        C["jev"],
        C["triple"],
        C["grade5"],
        C["text_image"],
        C["residual"],
        C["jev_json"],
        C["image"],
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("与人工 1–5 分的 Spearman（197 条）")
    ax.set_title("与人工 197 条档次的 Spearman")
    ax.grid(axis="x", color=C["line"], zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, value in enumerate(values):
        ax.text(value + 0.012, i, f"{value:.2f}", va="center", fontsize=9)
    _save(fig, "human_spearman.png")


def main() -> None:
    gate()
    grade_means()
    distributions()
    human_medians()
    pairwise()
    frozen_outcomes()
    human_spearman()
    print(OUT)


if __name__ == "__main__":
    main()
